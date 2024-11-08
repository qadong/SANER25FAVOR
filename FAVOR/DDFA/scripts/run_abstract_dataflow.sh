#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/absdf.log
#SBATCH --error=logs/absdf.log
#SBATCH --job-name="absdf"



PROJECT_ROOT="/home/l1/qingao/DeepDFA/DDFA"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 确保脚本在项目根目录下执行
cd "$PROJECT_ROOT"
source activate.sh

set -e

python -u sastvd/scripts/abstract_dataflow_full.py --workers 16 --no-cache --stage 1 $@
python -u sastvd/scripts/abstract_dataflow_full.py --workers 16 --no-cache --stage 2 $@
