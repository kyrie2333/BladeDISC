#!/bin/bash

# export PATH=/disc/software/nsight-system/bin:$PATH
export PATH=/opt/nvidia/nsight-compute/2023.1.0/:$PATH


     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
         apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-compute-2020.1.0 

     wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list && \
    apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nsight-systems nsight-compute
