PROJECT_ROOT="/home/l1/qingao/DeepDFA/DDFA"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 确保脚本在项目根目录下执行
cd "$PROJECT_ROOT"



python code_gnn/main_cli.py fit --config configs/config_bigvul.yaml --config configs/config_ggnn.yaml $@
