#!/bin/bash

# FILE=colreduct_naive
FILE=colreduct_transpose

nvcc -o ${FILE} ${FILE}.cu

CUDA_VISIBLE_DEVICES=4 ./${FILE}

# CUDA_VISIBLE_DEVICES=4 nsys profile -f true --stats=true -o ${FILE} ./${FILE}