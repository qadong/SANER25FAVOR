MODEL=Salesforce/codet5-base
export CUDA_VISIBLE_DEVICES=0 # 只使用编号为0的GPU

export HF_HOME=/home/hdd/qingao/cache/huggingface
export LD_LIBRARY_PATH=/root/miniconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib/:$LD_LIBRARY_PATH

python /home/l1/qingao/DeepDFA/CodeT5/run_gen.py \
    --do_train \
    --do_eval \
    --task repair \
    --sub_task none \
    --model_type codet5 \
    --data_num -1 \
    --num_train_epochs 30 \
    --warmup_steps 500 \
    --learning_rate 2e-5 \
    --patience 5 \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --data_dir None \
    --cache_path None \
    --output_dir /home/qingao/DeepDFA/CodeT5/saved_models/repair/flowgnn \
    --summary_dir tensorboard \
    --save_last_checkpoints \
    --res_dir /home/qingao/DeepDFA/CodeT5/saved_models/repair/flowgnn \
    --res_fn /home/qingao/DeepDFA/CodeT5/saved_models/repair/flowgnn/results/repair_codet5_base.txt \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_source_length 512 \
    --max_target_length 256 \
    --seed 4 \
    --flowgnn_data \
    --flowgnn_model
