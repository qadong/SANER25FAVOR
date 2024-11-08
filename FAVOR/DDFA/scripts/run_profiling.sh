#!/bin/bash
PROJECT_ROOT="/home/l1/qingao/DeepDFA/DDFA"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 确保脚本在项目根目录下执行
cd "$PROJECT_ROOT"


ckpt="$1"

for metric in profile time
do
    bash scripts/test.sh $ckpt --model.$metric True
done
