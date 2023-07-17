#!/bin/bash

export PATH=/disc/software/nsight-system/bin:$PATH
export PATH=/disc/software/nsight-compute:$PATH

# CUDA_VISIBLE_DEVICES=7 nsys profile --stats=true -w true -t cuda,nvtx,osrt,cudnn,cublas -c cudaProfilerApi \
#  -o breakdown --force-overwrite true ./gemm_a8w8

CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 2008 8192 6144
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 2008 2048 8192
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 2008 8192 10944
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 2008 5472 8192

CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 8000 8192 6144
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 8000 2048 8192
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 8000 8192 10944
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 8000 5472 8192

CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 16000 8192 6144
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 16000 2048 8192
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 16000 8192 10944
CUDA_VISIBLE_DEVICES=4,5,6,7 ./gemm_a8w8 16000 5472 8192

# nohup ./run.sh > run.log 2>&1 &