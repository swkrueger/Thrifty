#ifndef FASTCARD_WRAPPERS_H
#define FASTCARD_WRAPPERS_H

#include <complex>
#include <stdexcept>
#include <vector>

#include <volk/volk.h>

#define USE_FFTW
#include <fastcard/fastcard.h>
#include <fastcard/fargs.h>
#include <fastcard/fft.h>


class FastcardException: public std::exception {
  public:
    FastcardException(int code, std::string msg);
    virtual const char* what() const throw();
    int getCode();

  protected:
    std::string what_;
    int code_;
};

// wrapper around fastcard
class CarrierDetector {
  public:
    CarrierDetector(fargs_t *args);
    ~CarrierDetector();
    void start();
    bool next();
    void process();
    bool process_next();
    const fastcard_data_t& data();
    void cancel();
    void print_stats(FILE* out);

  private:
    fastcard_t* fastcard_;
    const fastcard_data_t* data_;
};

// Thin wrapper around volk_alloc and volk_free
// FIXME: there are better ways to do this, but who cares?
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

    // // Temporary helper functions
    // std::complex<float>* complex_data() {
    //     return reinterpret_cast<complex<float>*>(array_);
    // }
    // std::vector<std::complex<float>> complex_vector() {
    //     vector<complex<float>> vec;
    //     vec.assign(complex_data(), complex_data() + size_);
    //     return vec;
    // }

  private:
    T* array_ = NULL;
    size_t size_;
};

// Thin wrapper around fastcard fft
class FFT {
  public:
    FFT(size_t fft_len, bool forward);
    ~FFT();
    void execute();
    fcomplex* input() { return state_->input; }
    fcomplex* output() { return state_->output; }

  private:
    fft_state_t* state_ = NULL;
};

// Thin wrapper around FILE* to close it automatically
class CFile {
  public:
    CFile() : file_(NULL) {};
    CFile(std::string filename) : file_(NULL) { open(filename); };
    CFile(FILE* file) : file_(file) {};
    CFile(CFile&& other) : file_(other.file_) { other.file_ = NULL; }
    ~CFile() { close(); }
    void open(std::string filename);
    void open(FILE* file) { close(); file_ = file; };
    void flush();
    void close();
    void printf(const char* format, ...);
    FILE* file() { return file_; };

    // non-copyconstructible
    CFile(const CFile&) = delete;
    CFile& operator=(const CFile&) = delete;
  private:
    // TODO: use unique_ptr with deleter
    FILE* file_;
};

#endif /* FASTCARD_WRAPPERS_H */
