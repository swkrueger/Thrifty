#ifndef CORR_DETECTOR_H
#define CORR_DETECTOR_H

#include <vector>
#include <stddef.h>
#include <stdint.h>
#include <string>

// TODO: CorrDetector should not depend on fastcard
#include "fastcard_wrappers.h"

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

class CorrDetector {
  public:
    CorrDetector(const std::vector<float> &template_samples,
                 size_t block_len,
                 size_t history_len,
                 float corr_thresh_const,
                 float corr_thresh_snr);
    CorrDetection detect(const fastcard_data_t &carrier_det);

  protected:
    void set_template(const std::vector<float> &template_samples);
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

// TODO: move into some proper namespace
std::vector<float> load_template(std::string filename);

#endif /* CORR_DETECTOR_H */
