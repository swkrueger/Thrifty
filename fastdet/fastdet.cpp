// A quick-n-dirty proof-of-concept fast C++ implementation of Thrifty detect.
// This is a mess. This should be refactored.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <memory>
#include <inttypes.h>

#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>

#include <argp.h>

#include <volk/volk.h>

#define USE_FFTW
#include "fastcard.h"
#include "fargs.h"
#include "fft.h"
#include "parse.h"
#include "lib/base64.h"

using namespace std;


//// Fastcard wrapper
class FastcardException: public exception
{
  public:
    FastcardException(int code, string msg) : code_(code) {
        ostringstream ss;
        ss << "Fastcard error: return code " << code;
        if (msg.length() > 0) {
            ss << ". " << msg;
        }
        what_ = ss.str();
    };

    virtual const char* what() const throw() {
        return what_.c_str();
    }

    int getCode() { return code_; }

  protected:
    string what_;
    int code_;
};

// wrapper around fastcard
class CarrierDetector {
  private:
    fastcard_t* fastcard_;
    const fastcard_data_t* data_;

  public:
    CarrierDetector(fargs_t *args);
    ~CarrierDetector();
    void start();
    bool next();
    const fastcard_data_t& data();
    void cancel();
    void print_stats(FILE* out);
};

CarrierDetector::CarrierDetector(fargs_t *args) {
    // TODO: copy args
    fastcard_ = fastcard_new(args);
    if (fastcard_ == NULL) {
        throw FastcardException(-1, "failed to init fastcard");
    }
}

CarrierDetector::~CarrierDetector() {
    if (fastcard_) {
        fastcard_free(fastcard_);
    }
}

void CarrierDetector::start() {
    int ret = fastcard_start(fastcard_);
    if (ret != 0) {
        throw FastcardException(ret, "failed to start fastcard");
    }
}

bool CarrierDetector::next() {
    int ret = fastcard_next(fastcard_, &data_);
    if (ret != 0) {
        if (ret != 1) {
            throw FastcardException(ret, "reader stopped unexpectedly");
        }
        return false;
    }
    return true;
}

const fastcard_data_t& CarrierDetector::data() {
    return *data_;
}

void CarrierDetector::cancel() {
    fastcard_cancel(fastcard_);
}

void CarrierDetector::print_stats(FILE* out) {
    fastcard_print_stats(fastcard_, out);
}

//// Thin wrapper around volk_alloc and volk_free
// there are better ways to do this, but who cares?
template <class T>
class AlignedArray {
  public:
    AlignedArray(size_t size) : size_(size) { 
        size_t alignment = volk_get_alignment();  // TODO: cache alignment
        array_ = (T*)volk_malloc(size * sizeof(T), alignment);
        if (array_ == NULL) {
            throw std::runtime_error("volk_malloc failed");
        }
    }

    ~AlignedArray() {
        if (array_) {
            volk_free(array_);
        }
    }

    T* data() { return array_; }

    complex<float>* complex_data() {  // Temporary helper function
        return reinterpret_cast<complex<float>*>(array_);
    }

    vector<complex<float>> complex_vector() {  // Temporary helper function
        vector<complex<float>> vec;
        vec.assign(complex_data(), complex_data() + size_);
        return vec;
    }

  private:
    T* array_ = NULL;
    size_t size_;
};

//// Thin wrapper around fastcard fft
class FFT {
  public:
    FFT(size_t fft_len, bool forward) {
        state_ = fft_new(fft_len, forward);
        if (!state_) {
            throw std::runtime_error("Failed to init FFT");
        }
    }

    ~FFT() { if (state_) { fft_free(state_); } }
    void execute() { fft_perform(state_); }
    fcomplex* input() { return state_->input; }
    fcomplex* output() { return state_->output; }

  private:
    fft_state_t* state_ = NULL;
};

//// Correlation detector
struct CorrDetection {
    bool detected;
    uint16_t peak_idx;
    double peak_offset;
    float peak_power;
    float noise_power;
    float threshold;
    double carrier_offset;
};

void roll(fcomplex* output, const fcomplex* input, size_t len, int cnt) {
    size_t new_zero = (cnt < 0) ? (len + cnt) : cnt;
    memcpy(output + new_zero, input, (len - new_zero) * sizeof(fcomplex));
    memcpy(output, input + len - new_zero, (new_zero) * sizeof(fcomplex));
}

void print_fcomplex(fcomplex* array, size_t len) {
    while (len) {
        cout << array->real << "+j" << array->imag << " ";
        --len;
        ++array;
    }
    cout << endl;
}

class CorrDetector {
  public:
    CorrDetector(const vector<float> &template_samples,
                 size_t block_len,
                 size_t history_len,
                 float corr_thresh_const,
                 float corr_thresh_snr);
    CorrDetection detect(const fastcard_data_t &carrier_det);

  protected:
    void set_template(const vector<float> &template_samples);
    void set_window(size_t block_len, size_t history_len, size_t template_len);
    double interpolate_parabolic(float* peak_power);
    double interpolate_gaussian(float* peak_power);
    float estimate_noise(size_t peak_mag, float signal_energy);

  private:
    size_t len_;
    size_t corr_len_;
    float thresh_const_;
    float thresh_snr_;

    // volk aligned; calculated on init
    AlignedArray<fcomplex> template_fft_conj_;

    AlignedArray<fcomplex> shifted_fft_;
    AlignedArray<float> corr_power_;

    FFT ifft_;
    fcomplex* corr_fft_;  // owned by ifft_
    fcomplex* corr_;      // owned by ifft_

    size_t start_idx_;
    size_t stop_idx_;
    float template_energy_;
};

CorrDetector::CorrDetector(const vector<float> &template_samples,
                           size_t block_len,
                           size_t history_len,
                           float corr_thresh_const,
                           float corr_thresh_snr)
        : len_(block_len),
          corr_len_(block_len - template_samples.size() + 1),
          thresh_const_(corr_thresh_const),
          thresh_snr_(corr_thresh_snr),
          template_fft_conj_(block_len),
          shifted_fft_(block_len),
          corr_power_(corr_len_),
          ifft_(block_len, false) {

    set_template(template_samples);
    set_window(block_len, history_len, template_samples.size());
    corr_fft_ = ifft_.input();
    corr_ = ifft_.output();
}

void CorrDetector::set_template(const vector<float> &template_samples) {
    // calculate fft
    FFT template_fft_calc(len_, true);
    for (size_t i = 0; i < len_; ++i) {
        template_fft_calc.input()[i].real = template_samples[i];
        template_fft_calc.input()[i].imag = 0;
    }
    for (size_t i = template_samples.size(); i < len_; ++i) {
        template_fft_calc.input()[i].real = 0;
        template_fft_calc.input()[i].imag = 0;
    }
    template_fft_calc.execute();
    // calculate template conj
    volk_32fc_conjugate_32fc((lv_32fc_t*)template_fft_conj_.data(), (lv_32fc_t*)template_fft_calc.output(), len_);

    // calculate energy
    template_energy_ = 0;
    for (size_t i = 0; i < template_samples.size(); ++i) {
        template_energy_ += template_samples[i] * template_samples[i];
    }
}

void CorrDetector::set_window(
        size_t block_len,
        size_t history_len,
        size_t template_len) {

    assert(history_len >= template_len - 1);
    size_t padding = history_len - template_len + 1;
    size_t left_pad = padding / 2;
    size_t right_pad = padding-left_pad;

    size_t corr_len = block_len - template_len + 1;
    start_idx_ = left_pad;
    stop_idx_ = corr_len - right_pad;
}

double CorrDetector::interpolate_parabolic(float* peak_power) {
    // Apply parabolic interpolation to carrier / correlation peak.
    // Warning: we're not checking the boundaries!

    float a = sqrt((double)*(peak_power-1));
    float b = sqrt((double)*(peak_power));
    float c = sqrt((double)*(peak_power+1));
    float offset = (c - a) / (4*b - 2*a - 2*c);

    if (offset < -0.5) offset = -0.5;
    if (offset > 0.5) offset = 0.5;

    return offset;
}

double CorrDetector::interpolate_gaussian(float* peak_power) {
    // Apply parabolic interpolation to carrier / correlation peak.
    // WARNING: we're not checking the boundaries!

    double a = log(sqrt((double)*(peak_power-1)));
    double b = log(sqrt((double)*(peak_power)));
    double c = log(sqrt((double)*(peak_power+1)));
    double offset = (c - a) / (4*b - 2*a - 2*c);

    if (offset < -0.5) offset = -0.5;
    if (offset > 0.5) offset = 0.5;

    return offset;
}

float CorrDetector::estimate_noise(size_t peak_power, float signal_energy) {
    float signal_corr_energy = signal_energy * template_energy_;
    float noise_power = (signal_corr_energy - peak_power) / len_;
    if (noise_power < 0) {
        noise_power = 0;
    }
    return noise_power;
}

CorrDetection CorrDetector::detect(const fastcard_data_t &carrier_det) {
    // Frequency sync: roll
    roll(shifted_fft_.data(),
         carrier_det.fft,
         len_,
         -carrier_det.detection.argmax);

    // Calculate corr FFT
    volk_32fc_x2_multiply_32fc((lv_32fc_t*)corr_fft_,
                               (const lv_32fc_t*)shifted_fft_.data(),
                               (const lv_32fc_t*)template_fft_conj_.data(),
                               len_);
    // Calculate corr from FFT
    ifft_.execute();
    for (size_t i = 0; i < 2*corr_len_; ++i) {
        // normlize
        ((float*)corr_)[i] /= len_;
    }

    // Calculate magnitude
    volk_32fc_magnitude_squared_32f_a(corr_power_.data(),
                                      (const lv_32fc_t*)corr_,
                                      corr_len_);

    // Get peak
    uint16_t peak_idx;
    volk_32f_index_max_16u(
            &peak_idx,
            corr_power_.data() + start_idx_,
            stop_idx_ - start_idx_);
    peak_idx += start_idx_;
    float peak_power = corr_power_.data()[peak_idx];

    // Calculate threshold
    float signal_energy = carrier_det.detection.fft_sum / len_;
    float noise_power = estimate_noise(peak_power, signal_energy);
    float threshold = thresh_const_ + thresh_snr_ * noise_power;

    // Detection verdict
    bool detected = (peak_power > threshold);

    float* corr_power_peak = &corr_power_.data()[peak_idx];
    double offset = detected ? interpolate_gaussian(corr_power_peak) : 0;

    // Carrier interpolation
    double carrier_offset = interpolate_parabolic(
            &carrier_det.fft_power[carrier_det.detection.argmax]);

    CorrDetection det;
    det.detected = detected;
    det.peak_idx = peak_idx;
    det.peak_offset = offset;
    det.peak_power = *corr_power_peak;
    det.noise_power = noise_power;
    det.threshold = threshold;
    det.carrier_offset = carrier_offset;
    return det;
}


vector<float> load_template(string filename) {
    //  TODO: don't use native endianness, but standardize on
    //        either little or big endian
    ifstream ifs;
    std::ios_base::iostate exceptionMask = (ifs.exceptions() |
                                            std::ios::failbit |
                                            std::ios::badbit);
    ifs.exceptions(exceptionMask);

    try {
        ifs.open(filename);

        // read length
        uint16_t length;
        // TODO: proper error handling for read
        ifs.read((char*)&length, 2);

        // read data
        vector<float> data(length);
        ifs.read((char*)&data[0], length*sizeof(float));

        return data;

    } catch (std::ios_base::failure& e) {
        stringstream ss;
        ss << "Failed to load template: " << strerror(errno);
        throw std::runtime_error(ss.str());
    }
}

// Thin wrapper around FILE* to close it automatically
class CFile {
  public:
    CFile() : file_(NULL) {};
    CFile(std::string filename) : file_(NULL) { open(filename); };
    CFile(FILE* file) : file_(file) {};
    ~CFile() { close(); }
    void open(std::string filename);
    void open(FILE* file) { close(); file_ = file; };
    void flush();
    void close();
    void printf(const char* format, ...);
    FILE* file() { return file_; };
  private:
    FILE* file_;
};

void CFile::open(std::string filename) {
    close();
    if (filename.length() == 0) {
        file_ = NULL;
    } else if (filename == "") {
        file_ = stdout;
    } else {
        file_ = fopen(filename.c_str(), "w");
        if (file_ == NULL) {
            stringstream ss;
            ss << "Failed to open the file '" << filename << "': "
               << strerror(errno);
            throw std::runtime_error(ss.str());
        }
    }
}

void CFile::flush() {
    if (file_ != NULL) {
        fflush(file_);
    }
}

void CFile::printf(const char* format, ...) {
    if (file_ != NULL) {
        va_list args;
        va_start(args, format);
        vfprintf(file_, format, args);
        va_end(args);
    }
}

void CFile::close() {
    if (file_ != NULL && file_ != stdout) {
        fclose(file_);
    }
}


//// CLI stuff
// TODO: use proper command-line parser (e.g. tclap)

const char *argp_program_version = "fastdet " VERSION_STRING;
static const char doc[] = "FastDet: Fast Detector\n\n"
    "Like Thrifty, but faster.";

#define NUM_EXTRA_OPTIONS 6
static struct argp_option extra_options[] = {
    {"output", 'o', "<FILE>", 0,
        "Output card file\n('-' for stdout)\n[default: no output]", 1},
    {"card-output", 'x', "<FILE>", 0,
     "Write block to card file on detect\n('-' for stdout)\n[default: no output]", 1},

    // Correlator
    {0, 0, 0, 0, "Correlator settings:", 5},
    {"corr-threshold", 'u', "<constant>c<snr>s", 0,
        "Correlation detection theshold\n[default: 15s]", 5},
    {"template", 'z', "<FILE>", 0,
        "Load template from a .tpl file\n[default: template.tpl]", 5},
    {"rxid", 'r', "<int>", 0,
        "This receiver's unique identifier\n[default: -1]", 5}
};

unique_ptr<fargs_t, decltype(free)*> args = {NULL, free};
std::string output_file;
std::string card_output_file;
std::string template_file = "template.tpl";
float arg_corr_thresh_const = 0;
float arg_corr_thresh_snr = 15;
int rxid = -1;

static error_t parse_opt (int key, char *arg, struct argp_state *state) {
    if (key == 'o') {
        output_file = arg;
    } else if (key == 'x') {
        card_output_file = arg;
    } else if (key == 'u') {
        if (!parse_theshold_str(arg,
                                &arg_corr_thresh_const,
                                &arg_corr_thresh_snr)) {
            argp_usage(state);
        }
    } else if (key == 'z') {
        template_file = arg;
    } else if (key == 'r') {
        rxid = atoi(arg);
    } else if (key == ARGP_KEY_ARG) {
        // We don't take any arguments
        argp_usage(state);
    } else {
        int result = fargs_parse_opt(args.get(), key, arg);
        if (result == FARGS_UNKNOWN) {
            return ARGP_ERR_UNKNOWN;
        } else if (result == FARGS_INVALID_VALUE) {
            argp_usage(state);
        }
    }

    return 0;
}

std::unique_ptr<CarrierDetector> carrier_det;

void signal_handler(int signo) {
    (void)signo;  // unused
    if (carrier_det) {
        carrier_det->cancel();
    }
}


int main(int argc, char **argv) {
    // Argument parsing mess
    struct argp_option options[FARGS_NUM_OPTIONS + NUM_EXTRA_OPTIONS];
    memcpy(options,
           extra_options,
           sizeof(struct argp_option)*NUM_EXTRA_OPTIONS);
    memcpy(options + NUM_EXTRA_OPTIONS,
           fargs_options,
           sizeof(struct argp_option)*FARGS_NUM_OPTIONS);
    struct argp argp = {options, parse_opt, NULL,
                        doc, NULL, NULL, NULL};

    args.reset(fargs_new());
    argp_parse(&argp, argc, argv, 0, 0, 0);

    try {
        CFile out = CFile(output_file);
        CFile card = CFile(card_output_file);
        CFile info;
        if (!args->silent) {
            info.open((out.file() == stdout) ? stderr : stdout);
        }

        carrier_det.reset(new CarrierDetector(args.get()));
        vector<float> template_samples = load_template(template_file);
        CorrDetector corr_detect(template_samples,
                                 args->block_len,
                                 args->history_len,
                                 arg_corr_thresh_const,
                                 arg_corr_thresh_snr);

        vector<char> base64((2*args->block_len+2)/3*4 + 10);

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        signal(SIGQUIT, signal_handler);
        signal(SIGPIPE, signal_handler);

        // print header
        bool input_from_sdr = false;
        if (args->input_file) {
            input_from_sdr = (strcmp(args->input_file, "rtlsdr") == 0);
        }
        if (info.file() != NULL) {
            fargs_print_summary(args.get(), info.file(), input_from_sdr);
            info.printf("receiver id: %d\n", rxid);
            info.printf("corr threshold: constant = %g; snr = %g\n",
                       arg_corr_thresh_const, arg_corr_thresh_snr);
            info.printf("template: %s\n\n", template_file.c_str());
            info.flush();
        }

        if (card.file() != NULL && card.file() != stdout) {
            fargs_print_card_header(args.get(), card.file(),
                                    input_from_sdr, argp_program_version);
        }

        // Start detection!
        unsigned skip = args->skip;
        unsigned cnt = 0;
        if (skip > 0) {
            info.printf("\nSkipping %u block(s)... ", args->skip);
            info.flush();
        }

        carrier_det->start();

        while (carrier_det->next()) {
            if (skip > 0) {
                --skip;
                if (skip == 0) {
                    info.printf("done\n\n");
                    info.flush();
                }
                continue;
            }
            ++cnt;

            const fastcard_data_t& carrier = carrier_det->data();
            if (!carrier.detected) {
                continue;
            }

            // TODO: Don't block, but use a producer / consumer queue to
            // perform correlation detection async

            const CorrDetection corr = corr_detect.detect(carrier);

            int64_t block_idx = carrier.block->index;
            double soa = ((args->block_len - args->history_len) *
                          block_idx + corr.peak_idx) + corr.peak_offset;

            // output toad
            if (out.file() != NULL) {
                out.printf("%d %ld.%06ld %" PRId64 " %.8f"
                           " %u %.12f %f %f %u %f %f %f\n",
                           rxid,
                           carrier.block->timestamp.tv_sec,
                           carrier.block->timestamp.tv_usec,
                           carrier.block->index,
                           soa,
                           corr.peak_idx,
                           corr.peak_offset,
                           sqrt(corr.peak_power),
                           sqrt(corr.noise_power),
                           carrier.detection.argmax,
                           corr.carrier_offset,
                           sqrt(carrier.detection.max),
                           sqrt(carrier.detection.noise)
                           );
            }

            if (card.file() != NULL) {
                Base64encode(base64.data(),
                             (const char*) carrier.block->raw_samples,
                             args->block_len * 2);
                card.printf("%ld.%06ld %" PRId64 " %s\n",
                            carrier.block->timestamp.tv_sec,
                            carrier.block->timestamp.tv_usec,
                            carrier.block->index,
                            base64.data());
            }

            float carrier_snr_db = 10 * log10(carrier.detection.max /
                                              carrier.detection.noise);

            if (info.file() != NULL) {
                info.printf("block #%" PRId64 ": carrier @ %3u %+.1f = "
                            "%4.0f / %2.0f [>%2.0f] = %2.0f dB",
                            block_idx,
                            carrier.detection.argmax,
                            corr.carrier_offset,
                            sqrt(carrier.detection.max),
                            sqrt(carrier.detection.noise),
                            sqrt(carrier.detection.threshold),
                            carrier_snr_db);

                if (corr.detected) {
                    float corr_snr_db = 10 * log10(corr.peak_power /
                                                   corr.noise_power);
                    info.printf("; corr = %4.0f / %2.0f [>%2.0f] = %2.0f dB",
                                sqrt(corr.peak_power),
                                sqrt(corr.noise_power),
                                sqrt(corr.threshold),
                                corr_snr_db);
                }

                info.printf("\n");
            }
        }

        if (info.file() != NULL) {
            info.printf("\nRead %d blocks.\n", cnt);
            carrier_det->print_stats(info.file());
        }

    } catch (FastcardException& e) {
        cerr << e.what() << endl;
        return e.getCode();
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
