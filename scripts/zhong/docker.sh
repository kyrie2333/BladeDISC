#!/bin/bash

# sudo gpasswd -a zhongyun docker
# newgrp docker

# docker run -it --gpus all --name bladedisc-cu117 -v $PWD:/disc bladedisc/bladedisc:latest-devel-cu117 bash
# bladedisc/bladedisc:latest-devel-cu117

# docker exec -it --privileged bladedisc-cu117 bash
docker exec -it --privileged disc-cu113 bash
cd disc/BladeDISC/
PS1='${debian_chroot:+($debian_chroot)}\[\033[01;35;01m\]\u\[\033[00;00;01m\]@ \[\033[01;32;01m\]\w \[\033[00;00;01m\]\$ '
source /opt/venv_disc/bin/activate
python scripts/python/tao_build.py  /opt/venv_disc/  -s  test_tao_compiler

# export HTTPS_PROXY=http://blade_disc:r299eNZXVTUpuyg7@8.217.91.10:12357
# no_proxy=dl.google.com

# ssh-agent
# eval $(ssh-agent -s)
# ssh-keygen -o -t rsa -b 4096 -C "yunzhongOvO"
# ssh-add ~/.ssh/id_rsa 

# docker run  -it --gpus all --name disc-cu113 -v $PWD:/disc bladedisc/bladedisc:latest-devel-cu113 bash
# docker pull bladedisc/bladedisc:latest-devel-cu113

# nohup python scripts/python/tao_build.py  /opt/venv_disc/  -s  test_tao_compiler > compile.log 2>&1 &