cmake_minimum_required (VERSION 2.6)
add_library(gpu_fft mailbox.c gpu_fft.c gpu_fft_base.c gpu_fft_twiddles.c gpu_fft_shaders.c)
target_link_libraries (gpu_fft rt dl m)
