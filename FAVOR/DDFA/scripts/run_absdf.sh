#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/dbize.log
#SBATCH --error=logs/dbize.log
#SBATCH --job-name="dbize"

PROJECT_ROOT="/home/l1/qingao/DeepDFA/DDFA"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 确保脚本在项目根目录下执行
cd "$PROJECT_ROOT"

source activate.sh

python -u sastvd/scripts/dbize_absdf.py $@
