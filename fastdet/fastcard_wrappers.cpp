#include <stdarg.h>
#include <sstream>
#include <stdio.h>


#include "fastcard_wrappers.h"


using namespace std;

///
/// FastcardException
///
FastcardException::FastcardException(int code, string msg) : code_(code) {
    ostringstream ss;
    ss << "Fastcard error: return code " << code;
    if (msg.length() > 0) {
        ss << ". " << msg;
    }
    what_ = ss.str();
};

const char* FastcardException::what() const throw() { return what_.c_str(); }
int FastcardException::getCode() { return code_; }


///
/// CarrierDetector
///
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
    int ret = fastcard_next(fastcard_);
    if (ret != 0) {
        if (ret != 1) {
            throw FastcardException(ret, "reader stopped unexpectedly");
        }
        return false;
    }
    return true;
}

void CarrierDetector::process() {
    int ret = fastcard_process(fastcard_, &data_);
    if (ret != 0) {
        throw FastcardException(ret, "fastcard_process failed");
    }
}

bool CarrierDetector::process_next() {
    int ret = fastcard_process_next(fastcard_, &data_);
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


///
/// FFT
///
FFT::FFT(size_t fft_len, bool forward) {
    state_ = fft_new(fft_len, forward);
    if (!state_) {
        throw std::runtime_error("Failed to init FFT");
    }
}

FFT::~FFT() {
    if (state_) {
        fft_free(state_);
    }
}

void FFT::execute() {
    fft_perform(state_);
}


///
/// CFile
///
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
