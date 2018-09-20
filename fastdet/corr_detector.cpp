#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include <volk/volk.h>

#include "corr_detector.h"

using namespace std;

// Utility functions
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

///
/// CorrDetector
///
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
    for (size_t i = 0; i < template_samples.size(); ++i) {
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

    double a = sqrt((double)*(peak_power-1));
    double b = sqrt((double)*(peak_power));
    double c = sqrt((double)*(peak_power+1));
    double offset = (c - a) / (4*b - 2*a - 2*c);

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

CorrDetection CorrDetector::detect(const complex<float> *shifted_fft,
                                   float signal_energy) {
    // Note: shifted_fft_ should be memory-aligned

    // Calculate corr FFT
    volk_32fc_x2_multiply_32fc((lv_32fc_t*)corr_fft_,
                               (const lv_32fc_t*)shifted_fft,
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
            (uint16_t*)&peak_idx,
            corr_power_.data() + start_idx_,
            stop_idx_ - start_idx_);
    peak_idx += start_idx_;
    float peak_power = corr_power_.data()[peak_idx];

    // Calculate threshold
    float noise_power = estimate_noise(peak_power, signal_energy);
    float threshold = thresh_const_ + thresh_snr_ * noise_power;

    // Detection verdict
    bool detected = (peak_power > threshold);

    float* corr_power_peak = &corr_power_.data()[peak_idx];
    double offset = detected ? interpolate_gaussian(corr_power_peak) : 0;

    CorrDetection det;
    det.detected = detected;
    det.peak_idx = peak_idx;
    det.peak_offset = offset;
    det.peak_power = *corr_power_peak;
    det.noise_power = noise_power;
    det.threshold = threshold;
    return det;
}

CorrDetection CorrDetector::detect(const fastcard_data_t &carrier_det) {
    // Frequency sync: roll
    roll(shifted_fft_.data(),
         carrier_det.fft,
         len_,
         -carrier_det.detection.argmax);

    float signal_energy = carrier_det.detection.fft_sum / len_;

    // FIXME: casting madness
    CorrDetection det = detect((complex<float>*)shifted_fft_.data(), signal_energy);

    // Carrier interpolation
    // TODO: carrier interpolation should not be here!
    double carrier_offset = interpolate_parabolic(
            &carrier_det.fft_power[carrier_det.detection.argmax]);

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
