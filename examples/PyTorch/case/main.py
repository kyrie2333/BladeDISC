#!/usr/bin/python3
# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# export TORCH_BLADE_DEBUG_LOG=true 


import torch
import torch_blade
import torch_blade.clustering.support_fusion_group as fusion
import ctypes

_cudart = ctypes.CDLL('libcudart.so')


def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)
    

def func(x, y):
    x = (x - y) * x
    return torch.sum(x, dim=0)

with fusion.min_group_nodes(1):
    opt_func = torch.compile(backend='aot_disc')(func)

    # profile start
    cu_prof_start()
    opt_func(torch.rand(8192,2560).to('cuda'), torch.rand(8192,2560).to('cuda'))
    cu_prof_stop()
    # profile end

