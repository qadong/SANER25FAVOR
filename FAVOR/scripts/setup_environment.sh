#!/bin/bash
set -e

# 设置清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

# 创建并激活环境
conda env create -f ../environment.yml
conda activate deepdfa

# 设置环境变量
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/DDFA:$PYTHONPATH"
