#!/bin/bash

export PATH=/disc/cuda-11.4/nsight-systems-2021.2.4/bin:$PATH

# generate file ·breakdown.nsys-rep· with profiling information.
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -c cudaProfilerApi \
 -o bert_disc_opt_on --force-overwrite true python disc_bert.py

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -c cudaProfilerApi \
 -o bert_disc_opt_off --force-overwrite true python disc_bert.py

# extract profiling information into csv file.
nsys stats --report gpukernsum --format csv --force-overwrite true \
 -o bert_disc_opt_on bert_disc_opt_on.qdrep

nsys stats --report gpukernsum --format csv --force-overwrite true \
 -o bert_disc_opt_off bert_disc_opt_off.qdrep


python nsys_parse.py bert_disc_opt_on_gpukernsum.csv > disc_bert_on_new.txt

python nsys_parse.py bert_disc_opt_off_gpukernsum.csv > disc_bert_off_new.txt

bert_large_disc_opt_off