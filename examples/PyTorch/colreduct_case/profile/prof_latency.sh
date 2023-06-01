#!/bin/bash

export PATH=/disc/software/nsight-system/bin:$PATH
export PATH=/disc/software/nsight-compute:$PATH

# generate file ·breakdown.nsys-rep· with profiling information.
nsys profile --stats=true -w true -t cuda,nvtx,osrt,cudnn,cublas -c cudaProfilerApi \
 -o breakdown --force-overwrite true python ../main.py
# extract profiling information into csv file.
nsys stats --report gpukernsum --format csv --force-overwrite true \
 -o breakdown breakdown.nsys-rep

# /disc/software/nsight-compute/ncu -o breakdown -f --csv /usr/bin/python /disc/BladeDISC/examples/PyTorch/case/main.py 

# /opt/nvidia/nsight-compute/2023.1.0/ncu -o breakdown -f  --csv --set full /usr/bin/python ../main.py 

CUDA_VISIBLE_DEVICES=0 /opt/nvidia/nsight-compute/2023.1.0/ncu -o colreduct -f --target-processes all --rule SOLBottleneck --set full --profile-from-start off /usr/bin/python ../main.py 