#!/bin/bash

PROJECT_ROOT="/home/l1/qingao/DeepDFA/DDFA"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 确保脚本在项目根目录下执行
cd "$PROJECT_ROOT"

for i in $(seq 0 4)
do
    echo training on mixed-project fold $fold
    bash scripts/train.sh --data.split cross_project_fold_${fold}_dataset
    ckpt_dir=$(ls -td lightning_logs/* | tac | tail -n1)/checkpoints
    ckpt=$(ls $ckpt_dir/performance-*.ckpt)
    echo evaluating fold $fold on ckpt $ckpt
    bash scripts/test.sh $ckpt --data.split cross_project_fold_${fold}_dataset
    bash scripts/test.sh $ckpt --data.split cross_project_fold_${fold}_holdout
done
