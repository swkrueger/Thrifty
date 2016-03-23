fastcard: fastcard.c Makefile
	gcc -std=c99 -O3 fastcard.c lib/base64.c -o fastcard -lfftw3f -lm -lvolk

GPUFFT_SRC = mailbox.c gpu_fft.c gpu_fft_base.c gpu_fft_twiddles.c gpu_fft_shaders.c
GPUFFT_PATH = $(GPUFFT_SRC:%=lib/gpu_fft/%)
fastcard_pi: fastcard.c Makefile
	gcc -std=c99 -O3 fastcard.c lib/base64.c $(GPUFFT_PATH) -o fastcard_pi -lm -lvolk -lrt -ldl -march=armv6 -mfloat-abi=hard -mfpu=vfp

generate: generate.cpp
	g++ generate.cpp -o gen

