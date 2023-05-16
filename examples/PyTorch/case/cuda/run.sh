#!/bin/bash

# FILE=colreduct1
FILE=colreduct2.5
# FILE=colreduct3

nvcc -o ${FILE} ${FILE}.cu

CUDA_VISIBLE_DEVICES=5 ./${FILE}

# CUDA_VISIBLE_DEVICES=4 nsys profile -f true --stats=true -o ${FILE} ./${FILE}
# CUDA_VISIBLE_DEVICES=4 ncu -o colreduct2 \
# -f --set full --rule SOLBottleneck --target-processes all \
#  ./colreduct2
