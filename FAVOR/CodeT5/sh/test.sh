MODEL=Salesforce/codet5-base
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/hdd/qingao/cache/huggingface

python /home/l1/qingao/DeepDFA/CodeT5/run_gen.py \
    --do_test \
    --task repair \
    --sub_task none \
    --model_type codet5 \
    --data_num -1 \
    --num_train_epochs 30 \
    --warmup_steps 500 \
    --learning_rate 2e-4 \
    --patience 5 \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --data_dir None \
    --cache_path None \
    --output_dir /home/hdd/qingao/DeepDFA/CodeT5/saved_models/repair/codeT5/onlyt5 \
    --model_path /home/hdd/qingao/DeepDFA/CodeT5/saved_models/repair/codeT5/checkpoint-best-acc/pytorch_model.bin \
    --summary_dir tensorboard \
    --save_last_checkpoints \
    --res_dir /home/hdd/qingao/DeepDFA/CodeT5/saved_models/repair/codeT5 \
    --res_fn /home/hdd/qingao/DeepDFA/CodeT5/saved_models/repair/codeT5/repair_codet5_base.txt \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_source_length 512 \
    --max_target_length 256 \
    --seed 4 \
    # --flowgnn_data \
    # --flowgnn_model
