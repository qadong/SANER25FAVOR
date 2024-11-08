#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=1-00:00:00
#SBATCH --mem=32GB
#SBATCH --array=0-99%10
#SBATCH --err="logs/getgraphs_%A_%a.out"
#SBATCH --output="logs/getgraphs_%A_%a.out"
#SBATCH --job-name="getgraphs"

PROJECT_ROOT="/home/l1/qingao/DeepDFA/DDFA"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 确保脚本在项目根目录下执行
cd "$PROJECT_ROOT"

source activate.sh

if [ ! -z "$SLURM_ARRAY_TASK_ID"]
then
    jan="--job_array_number $SLURM_ARRAY_TASK_ID"
else
    jan=""
fi

# Start singularity instance
python -u sastvd/scripts/getgraphs.py bigvul --sess $jan --num_jobs 100 --overwrite $@
