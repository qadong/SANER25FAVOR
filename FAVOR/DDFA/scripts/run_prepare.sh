# 设置项目的根目录路径
PROJECT_ROOT="/home/l1/qingao/DeepDFA/DDFA"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 确保脚本在项目根目录下执行
cd "$PROJECT_ROOT"

source activate.sh

# Start singularity instance
python -u sastvd/scripts/prepare.py --dataset bigvul --global_workers 12 $@
