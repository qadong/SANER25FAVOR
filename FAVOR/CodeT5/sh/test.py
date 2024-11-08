import itertools
import subprocess
import os
# Parameter values to test

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_HOME'] = '/home/hdd/qingao/cache/huggingface'
knn_temp_values = [60, 80]
lmbda_values = [0.2, 0.4, 0.6]
k_values = [16, 32, 64]

# Log file to store results
log_file = '/home/l1/qingao/DeepDFA/CodeT5/log.txt'

# Base command
base_command = [
    'python', '/home/l1/qingao/DeepDFA/CodeT5/run_gen.py',
    '--do_test',
    '--task', 'repair',
    '--pattern_store',
    '--sub_task', 'none',
    '--model_type', 'codet5',
    '--data_num', '-1',
    '--num_train_epochs', '30',
    '--warmup_steps', '500',
    '--learning_rate', '2e-4',
    '--patience', '5',
    '--tokenizer_name', 'Salesforce/codet5-base',
    '--model_name_or_path', 'Salesforce/codet5-base',
    '--data_dir', 'None',
    '--cache_path', 'None',
    '--output_dir', '/home/hdd/qingao/DeepDFA/CodeT5/saved_models/repair/codeT5/remota',
    '--summary_dir', 'tensorboard',
    '--save_last_checkpoints',
    '--res_dir', '/home/hdd/qingao/DeepDFA/CodeT5/saved_models/repair/codeT5',
    '--res_fn', '/home/hdd/qingao/DeepDFA/CodeT5/saved_models/repair/codeT5/repair_codet5_base.txt',
    '--train_batch_size', '16',
    '--eval_batch_size', '8',
    '--max_source_length', '512',
    '--max_target_length', '256',
    '--model_path', '/home/l1/qingao/DeepDFA/CodeT5/home/hdd/qingao/RMGArepair/saved_model/step5_nodes200_dec_enc_dec_gnnf_wd1e-4/pytorch_model.bin',
    '--seed', '4',
    '--flowgnn_data',
    '--flowgnn_model',
    '--knn_gpu',
    '--dstore_dir', 'knn_cache',
    '--dstore_size', '507357',
    '--use_pattern',
    '--retomaton'
]

# Generate all combinations of parameters
combinations = list(itertools.product(knn_temp_values, lmbda_values, k_values))

# Function to run the experiment
def run_experiment(knn_temp, lmbda, k):
    command = base_command + [
        '--knn_temp', str(knn_temp),
        '--lmbda', str(lmbda),
        '--k', str(k)
    ]
    
    # Run the command and capture output
    with open(log_file, 'a') as log:
        log.write(f'Running experiment with knn_temp={knn_temp}, lmbda={lmbda}, k={k}\n')
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(result.stdout + '\n')
        log.write('-' * 80 + '\n')

# Run all combinations of parameters
for knn_temp, lmbda, k in combinations:
    print(f'test {knn_temp} {lmbda} {k} begin')
    run_experiment(knn_temp, lmbda, k)

print("All experiments are completed. Results are stored in log.txt")
