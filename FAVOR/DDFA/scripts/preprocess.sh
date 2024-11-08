#!/bin/bash

PROJECT_ROOT="/home/l1/qingao/DeepDFA/DDFA"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 确保脚本在项目根目录下执行
cd "$PROJECT_ROOT"

set -e




bash scripts/run_prepare.sh $@
bash scripts/run_getgraphs.sh $@ # Make sure Joern is installed!
bash scripts/run_dbize.sh $@
bash scripts/run_abstract_dataflow.sh $@
bash scripts/run_absdf.sh $@
