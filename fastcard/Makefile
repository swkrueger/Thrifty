fastcard: fastcard.c Makefile
	gcc -std=c99 -O3 fastcard.c lib/base64.c -o fastcard -lfftw3f -lm -lvolk

HELLOC = mailbox.c gpu_fft.c gpu_fft_base.c gpu_fft_twiddles.c gpu_fft_shaders.c
HELLOC_PATH = $(HELLOC:%=lib/hello_fft/%)
fastcard_pi: fastcard.c Makefile
	gcc -std=c99 -O3 fastcard.c lib/base64.c $(HELLOC_PATH) -o fastcard_pi -lm -lvolk -lrt -ldl -march=armv6 -mfloat-abi=hard -mfpu=vfp

generate: generate.cpp
	g++ generate.cpp -o gen

