#!/bin/bash

# FILE=colreduct1
# FILE=colreduct2
# FILE=colreduct3
# FILE=colreduct_disc
FILE=colreduct_all

nvcc -o ${FILE} ${FILE}.cu

# ./${FILE} 8192 8192
CUDA_VISIBLE_DEVICES=4 ./${FILE} 512 512

CUDA_VISIBLE_DEVICES=4 nsys profile -f true --stats=true -o ${FILE} ./${FILE} 512 512
# CUDA_VISIBLE_DEVICES=4 ncu -o colreduct2 \
# -f --set full --rule SOLBottleneck --target-processes all \
#  ./colreduct2

# nohup python3 profile_schedule_bk1.py > profile_schedule_bk1_1.log 2>&1 &
# nohup python3 profile_schedule_bk2.py > profile_schedule_bk2.log 2>&1 &
nohup  python3 compare_schedule_disc.py > compare_schedule_disc.log 2>&1 &
