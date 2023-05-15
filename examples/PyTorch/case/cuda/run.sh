#!/bin/bash

FILE=colreduct2_32x8
# FILE=colreduct3

nvcc -o ${FILE} ${FILE}.cu

CUDA_VISIBLE_DEVICES=5 ./${FILE}

# CUDA_VISIBLE_DEVICES=4 nsys profile -f true --stats=true -o ${FILE} ./${FILE}