#!/usr/bin/python

import numpy as np
import csv
import sys

def is_compute_intensive(kernel_name: str):
  keywords = ['gemm', 'gemv', 'cudnn', 'cublas', 'cutlass', 'conv1', 'conv2',
              'matmul']
  for key in keywords:
    if key in kernel_name:
      return True
  return False


def analyze_nsys_csv(file: str):
  entries = []
  with open(file) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
      if "Time (%)" in row[0]:
        continue
      # name, time, count
      entries.append([row[8], row[1], row[2]])
  
  comp_intensive_entries = []
  mem_intensive_entries = []

  for e in entries:
    kernel_name = e[0]
    if is_compute_intensive(kernel_name):
      comp_intensive_entries.append(list(e))
    else:
      mem_intensive_entries.append(list(e))
  
  print('Sumary.\n')
  mem_time = float(np.sum([int(e[1]) for e in mem_intensive_entries])) / 1e6
  mem_count = np.sum([int(e[2]) for e in mem_intensive_entries])
  print('memory-intensive ops:\n')
  print(f'\t total time: {mem_time} ms, kernel number: {mem_count}\n')
  for e in mem_intensive_entries:
    print(f'\t\t {e[0]}, {float(e[1]) / 1e6} ms, {e[2]}')

  print('\n\n')

  comp_time = float(np.sum([int(e[1]) for e in comp_intensive_entries])) / 1e6
  comp_count = np.sum([int(e[2]) for e in comp_intensive_entries])
  print('compute-intensive ops:\n')
  print(f'\t total time: {comp_time} ms, kernel number: {comp_count}\n')
  for e in comp_intensive_entries:
    print(f'\t\t {e[0]}, {float(e[1]) / 1e6} ms, {e[2]}')


if __name__ == '__main__':
  if (len(sys.argv) != 2):
    print('Usage:\n\tpython path/to/this/script path/to/nvprof/log/file')
    exit(1)

  analyze_nsys_csv(sys.argv[1])