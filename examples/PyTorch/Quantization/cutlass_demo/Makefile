ARCH=80
GENCODE_FLAGS=-gencode arch=compute_${ARCH},code=sm_${ARCH}
CUTLASS_ROOT=/disc/software/cutlass
FLAGS=${GENCODE_FLAGS} -O3 --std=c++17 -m64 -lcuda -I ${CUTLASS_ROOT}/include -I ${CUTLASS_ROOT}/tools/util/include -I ${CUTLASS_ROOT}/tools/library/include -I ${CUTLASS_ROOT}/examples/common -I .

all: gemm_a8w8 \
	 demo
	 
demo: demo.cu
	nvcc ${FLAGS} -o demo demo.cu

gemm_a8w8: gemm_a8w8.cu
	nvcc ${FLAGS} -o gemm_a8w8 gemm_a8w8.cu

run: gemm_a8w8
	./gemm_a8w8

clean:
	rm demo
	rm gemm_a8w8