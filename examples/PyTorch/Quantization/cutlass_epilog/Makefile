ARCH=86
GENCODE_FLAGS=-gencode arch=compute_${ARCH},code=sm_${ARCH}
CUTLASS_ROOT=/home/zhengzhen/workspace/release/cutlass
FLAGS=${GENCODE_FLAGS} -O3 --std=c++17 -m64 -lcuda -I ${CUTLASS_ROOT}/include -I ${CUTLASS_ROOT}/tools/util/include -I ${CUTLASS_ROOT}/tools/library/include -I ${CUTLASS_ROOT}/examples/common -I .

all: gemm_elemwise_epilogue

gemm_elemwise_epilogue: gemm_elemwise_epilogue.cu
	nvcc ${FLAGS} -o gemm_elemwise_epilogue gemm_elemwise_epilogue.cu

run: gemm_elemwise_epilogue
	./gemm_elemwise_epilogue

clean:
	rm gemm_elemwise_epilogue