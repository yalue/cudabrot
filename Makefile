.PHONY: all clean hip

CFLAGS := -Wall -Werror -O3 -g -fPIC

NVCCFLAGS := -g --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--cudart=shared -arch=compute_30 \
	--generate-code arch=compute_50,code=[compute_50,sm_50] \
	--generate-code arch=compute_62,code=[compute_62,sm_62]

all: cudabrot

cudabrot: cudabrot.cu
	nvcc $(NVCCFLAGS) -o cudabrot cudabrot.cu -lm

hip: cudabrot.cu
	hipify-perl cudabrot.cu > cudabrot_hip.cpp
	hipcc $(CFLAGS) \
		-I/opt/rocm/hiprand/include \
		-I/opt/rocm/rocrand/include \
		-o cudabrot \
		cudabrot_hip.cpp

clean:
	rm -f cudabrot

