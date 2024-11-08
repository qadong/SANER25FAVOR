# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import knnlm as knnlm
import retomaton as retomaton
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import re
import csv
import torch
import json
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
import pandas as pd
# from evaluator import smooth_bleu
# from evaluator.CodeBLEU import calc_code_bleu
# from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_data
from configs import add_args, set_seed, set_dist
from flowT5 import FLOWT5
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.replace(' ','')
    tokens = tokens.replace("</s>", "")
    tokens = tokens.replace("<start>", "").replace('<end>','')
    tokens = re.sub(r'\s+', ' ', tokens)
    tokens = tokens.strip()
    return tokens



def eval_acc_epoch(args, eval_dataloader, model, tokenizer,flowgnn_dataset):
    # Start evaluating model
    logger.info("  " + "***** Running acc evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    exact_match = 0
    num_missing = 0
    total_num = 0

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        input_ids = batch['input_ids'].to('cuda:0')
        attention_mask = batch['attention_mask'].to('cuda:0')
        labels = batch['labels'].to('cuda:0')
        index = batch['index'].to('cuda:0')
        if flowgnn_dataset is None:
            graphs = None
        else:
            graphs, keep_idx = flowgnn_dataset.get_indices(index)

            num_missing_batch = len(index) - len(keep_idx)
            num_missing += num_missing_batch
            input_ids = input_ids[keep_idx]
            target_ids = target_ids[keep_idx]
            attention_mask = attention_mask[keep_idx]
            index = index[keep_idx]
            if num_missing_batch > 0:
                logger.info("%d examples missing in batch", num_missing_batch)
            if graphs is None:
                print('graphs is None')
                logger.info("skipping batch of %d items, graphs is None for indices: %s", len(index), index)
                continue


        with torch.no_grad():
            if args.model_type == 'roberta':
                assert NotImplementedError
            else:
                # print('input_ids',input_ids.size())
                # print('attention_mask',attention_mask.size())
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                graph = graphs if graphs is not None else None,
                                max_length=args.max_target_length)
            for output, target_id in zip(outputs, labels):
                decoded_output = tokenizer.decode(output.tolist(), skip_special_tokens=True)
                decoded_target = tokenizer.decode(target_id.tolist(), skip_special_tokens=True)
                if clean_tokens(decoded_output) == clean_tokens(decoded_target):
                    exact_match += 1
                total_num += 1
    eval_acc = exact_match / total_num


    return eval_acc


def test_acc_epoch(args, eval_dataloader, model, tokenizer, graphs_by_id):
    # Start evaluating model
    logger.info("  " + "***** Running acc evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    exact_match = 0
    num_missing = 0
    total_num = 0

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  

    outputs = []
    all_results = []  
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="test ppl"):
        input_ids = batch['input_ids'].to('cuda:0')
        attention_mask = batch['attention_mask'].to('cuda:0')
        labels = batch['labels'].to('cuda:0')
        index = batch['index'].to('cuda:0')

        index_list = index.tolist()
        if graphs_by_id is None:
            graphs = None
        else:
            graphs = [graphs_by_id[i].to('cuda:0') for i in index_list if i in graphs_by_id]


        with torch.no_grad():
            if args.model_type == 'roberta':
                assert NotImplementedError
            else:
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                         graph=graphs if graphs is not None else None,
                                         max_length=args.max_target_length)

            for output, target_id, idx in zip(outputs, labels, index):

                match = False
                decoded_output = tokenizer.decode(output.tolist(), skip_special_tokens=True)
                decoded_target = tokenizer.decode(target_id.tolist(), skip_special_tokens=True)

                # Clean and compare tokens for exact match
                if clean_tokens(decoded_output) == clean_tokens(decoded_target):
                    exact_match += 1
                    match = True
                total_num += 1

     
                result = {
                    "index": idx.item(),
                    "output": decoded_output,
                    "label": decoded_target,
                    "match": match
                }
                all_results.append(result)


    output_file = os.path.join(output_dir, f"{args.k}_{args.lmbda}_{args.knn_temp}_eval_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f'successfully save result to {output_file}')
    test_acc = exact_match / total_num
    return test_acc

def main():
    import transformers
    import torch as th
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    # config, code_model, tokenizer = build_or_load_gen_model(args)
    model_name = 'Salesforce/codet5-base'
    tokenizer = AutoTokenizer.from_pretrained(
    '/home/hdd/qingao/cache/huggingface/transformers/models--Salesforce--codet5-base/snapshots/4078456db09ba972a3532827a0b5df4da172323c'
    )
    tokenizer.add_tokens(["Vul_Start","Vul_End"])
    code_model = transformers.T5ForConditionalGeneration.from_pretrained(
    '/home/hdd/qingao/cache/huggingface/transformers/models--Salesforce--codet5-base/snapshots/4078456db09ba972a3532827a0b5df4da172323c'
    )
    code_model.to('cuda:0')
    args.device = code_model.device
    code_model.resize_token_embeddings(len(tokenizer))
    # if args.n_gpu > 1:
    #     # for DataParallel
    #     model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'repair.log'), 'w')

    ### Graph data
    if args.flowgnn_data:
        from dgl.data.utils import load_graphs
        allfeats = [
            "api", "datatype", "literal", "operator",
        ]
        graphs, graph_labels = load_graphs('/home/l1/qingao/graphs_test200.bin')
        graphs_by_id = dict(zip(graph_labels["graph_ids"].tolist(), graphs))

        def load_additional_features(graphs_by_id, feat_names, split="fixed", sample_text=""):
           
            for feat in feat_names:
                
                prefix = "_ABS_DATAFLOW_"
                filepath = f"/home/l1/qingao/DeepDFA/DDFA/storage/processed/bigvul/nodes_feat_{prefix}{feat}_all_limitall_10000_limitsubkeys_10000_{split}{sample_text}.csv"
           
                feat_df = pd.read_csv(filepath, index_col=0)
                
         
                for graph_id, group in tqdm(feat_df.groupby("graph_id"), f"Adding feature {feat}"):
                    if graph_id in graphs_by_id:
                        g = graphs_by_id[graph_id]
                      
                        feat_column = next(c for c in feat_df.columns if c.startswith(f"_ABS_DATAFLOW_{feat}"))
                        
                        g.ndata[f"_ABS_DATAFLOW_{feat}"] = th.LongTensor(group[feat_column].tolist())
                        
            return graphs_by_id

        graphs_by_id = load_additional_features(graphs_by_id, allfeats)


    else:
        graphs_by_id = None

    if args.flowgnn_model:
        import sys
        sys.path.append("/home/l1/qingao/DeepDFA")
        from CodeT5.flowgnn import FlowGNNGGNNModule
        logger.info("ACTIVATING FLOWGNN MODEL")
        # input_dim = flowgnn_datamodule.input_dim
        input_dim = 8
        feat = "_ABS_DATAFLOW_datatype_all_limitall_1000_limitsubkeys_1000"
        gtype = "cfg"
        label_style = "graph"
        dsname = "bigvul"
        node_type_feat = None
        concat_all_absdf = True
        hidden_dim = 64
        n_steps = 5
        num_output_layers = 3
        
        flowgnn_model = FlowGNNGGNNModule(
            feat,
            input_dim,
            hidden_dim,
            n_steps,
            num_output_layers,
            label_style=label_style,
            concat_all_absdf=concat_all_absdf,
            encoder_mode=True,
        )
        logger.info("FlowGNN output dim: %d", flowgnn_model.out_dim)
    else:
        flowgnn_model = None 

    if args.flowgnn_model:
        config = code_model.config
        model = FLOWT5(config,flow_gnn=flowgnn_model)

    else:
        model = code_model
    model = model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    train_data = []
    df = pd.read_csv("../valid_index.csv", skip_blank_lines=True)


    print("Columns in the CSV file:", df.columns)

    valid_index = df['graph_id'].tolist()
    print(len(valid_index))
    with open('/home/l1/qingao/DeepDFA/DDFA/storage/external/train_data.json', 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    val_data = []
    with open('/home/l1/qingao/DeepDFA/DDFA/storage/external/val_data.json', 'r') as f:
        for line in f:
            val_data.append(json.loads(line))

    test_data = []
    with open('/home/l1/qingao/DeepDFA/DDFA/storage/external/test_data.json', 'r') as f:
        for line in f:
            test_data.append(json.loads(line))


    filtered_data = [item for item in train_data if item.get('Unnamed: 0') in valid_index]
    filtered_val_data = [item for item in val_data if item.get('Unnamed: 0') in valid_index]
    filtered_test_data = [item for item in test_data if item.get('Unnamed: 0') in valid_index]
    print('filtered_test_data:',len(filtered_test_data))
    train_df = pd.DataFrame(filtered_data)  
    val_df = pd.DataFrame(filtered_val_data)
    test_df = pd.DataFrame(filtered_test_data) 

    train_Ds = Dataset.from_pandas(train_df)
    val_Ds = Dataset.from_pandas(val_df)
    test_Ds = Dataset.from_pandas(test_df) 
    def process_func(example):
        MAX_LENGTH = 512
        source = ''.join(example['source']) if isinstance(example['source'], list) else example['source']
        target = ''.join(example['target']) if isinstance(example['target'], list) else example['target']
        inputs = tokenizer(source, truncation=True, max_length=MAX_LENGTH, padding='max_length')
        labels = tokenizer(target, truncation=True, max_length=MAX_LENGTH, padding='max_length')
        index = example['Unnamed: 0']
        return {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "labels": labels['input_ids'],
            "index" : torch.tensor(index)
        }

    train_ds = train_Ds.map(process_func, batched=False, remove_columns=train_Ds.column_names)
    val_ds = val_Ds.map(process_func, batched=False, remove_columns=val_Ds.column_names)
    test_ds = test_Ds.map(process_func, batched=False, remove_columns=test_Ds.column_names)
    # 输出筛选后的数据
    train_dataloader = DataLoader(train_ds, collate_fn=default_data_collator, batch_size=16, pin_memory=True)
    eval_dataloader = DataLoader(val_ds, collate_fn=default_data_collator, batch_size=1, pin_memory=True)
    test_dataloader = DataLoader(test_ds, collate_fn=default_data_collator, batch_size=8, pin_memory=True)
    if args.do_train:



        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}

        ### change 
        # global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        
        global_step ,best_acc, best_loss = 0, -1, 1e6


        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        num_missing = 0



        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                input_ids = batch['input_ids'].to('cuda:0')
                attention_mask = batch['attention_mask'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')
                index = batch['index'].to('cuda:0')

                if flowgnn_dataset is None:
                    graphs = None
                else:
                    graphs, keep_idx = flowgnn_dataset.get_indices(index)

                    num_missing_batch = len(index) - len(keep_idx)
                    num_missing += num_missing_batch
                    # print('input_ids', input_ids.size())
                    input_ids = input_ids[keep_idx]
                    # print('input_ids', input_ids.size())
                    attention_mask = attention_mask[keep_idx]
                    target_ids = target_ids[keep_idx]
                    index = index[keep_idx]


                    if num_missing_batch > 0:
                        logger.info("%d examples missing in batch", num_missing_batch)
                    if graphs is None:
                        logger.info("skipping batch of %d items, graphs is None for indices: %s", len(index), index)
                        continue
                if args.model_type == 'roberta':
                    loss, _, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                       target_ids=target_ids, graph=graphs)
                else:
                    if args.flowgnn_model:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    labels=target_ids,
                                    graph=graphs if graphs else None)
                        # outputs = model(
                        #         input_ids=input_ids,
                        #         attention_mask=input_ids,
                        #         decoder_input_ids=target_ids,
                        #         decoder_attention_mask=target_mask,
                        #         graph=graphs
                        #     )
                    else:
                        outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))



            if args.do_eval:
                # Eval model with dev dataset
                eval_acc=eval_acc_epoch(args, eval_dataloader, model, tokenizer,flowgnn_dataset=flowgnn_dataset if flowgnn_dataset else None)

                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_acc}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_acc, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_acc > best_acc:
                    not_loss_dec_cnt = 0
                    logger.info("  Best acc:%s", eval_acc)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_acc))
                    best_acc = eval_acc

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best acc model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        model_path = args.model_path
        logger.info("Reload model from {}".format(model_path))
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda:0')))
        
        def store(args, eval_dataloader, model, graphs_by_id):
            # Start evaluating model
            logger.info("  " + "***** Running acc evaluation *****")
            logger.info("  Num examples = %d", len(eval_dataloader))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="test ppl"):
                input_ids = batch['input_ids'].to('cuda:0')
                attention_mask = batch['attention_mask'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')
                index = batch['index'].to('cuda:0')

                index_list = index.tolist()
                if graphs_by_id is None:
                    graphs = None
                else:
                    graphs = [graphs_by_id[i].to('cuda:0') for i in index_list if i in graphs_by_id]


                with torch.no_grad():
                    if args.model_type == 'roberta':
                        assert NotImplementedError
                    else:
                        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                                labels = labels,
                                                graph=graphs if graphs is not None else None)
 


        if args.pattern_store:
            if args.retomaton or args.cluster_dstore:
                print('***************************using retomaton')
                knn_wrapper = retomaton.RetomatonWrapper(dstore_size=args.dstore_size, dstore_dir=args.dstore_dir, 
                    dimension=args.dimension, 
                    knn_sim_func=args.knn_sim_func, knn_keytype=args.knn_keytype,
                    no_load_keys=args.no_load_keys, move_dstore_to_mem=args.move_dstore_to_mem, knn_gpu=args.knn_gpu,
                    recompute_dists=args.recompute_dists,
                    k=args.k, lmbda=args.lmbda, knn_temp=args.knn_temp, probe=args.probe,
                    no_pointer=args.no_pointer, min_knns=args.min_knns, max_knns=args.max_knns,
                    members=args.members)
            elif args.store_knn == True or args.build_index == True:
                knn_wrapper = knnlm.KNNSaver(dstore_size=args.dstore_size, dstore_dir=args.dstore_dir, 
                    tokenizer=tokenizer,
                    dimension=args.dimension, knn_keytype=args.knn_keytype)
            else:
                knn_wrapper = knnlm.KNNWrapper(dstore_size=args.dstore_size, dstore_dir=args.dstore_dir, 
                dimension= args.dimension, 
                knn_sim_func=args.knn_sim_func, knn_keytype=args.knn_keytype,
                no_load_keys=args.no_load_keys, move_dstore_to_mem=args.move_dstore_to_mem, knn_gpu=args.knn_gpu,
                recompute_dists=args.recompute_dists,
                k=args.k, lmbda=args.lmbda, knn_temp=args.knn_temp, probe=args.probe)

            if knn_wrapper is not None:
                knn_wrapper.break_into(model)
                print('******************************break-into')
                logger.info('******************************break-into')

            
            if args.store_knn:
                total_eval_tokens = 0    
                for data in train_data:
                    target = data ['target']
                    chunk = tokenizer.encode(target,max_length=512)
                    total_eval_tokens += len([x for x in chunk[0:] if x != -100])
                print('total_eval_tokens:',total_eval_tokens)
                store(args, train_dataloader, model, graphs_by_id)
                print('successfully store pattern store!!!')
            if args.build_index:
                knn_wrapper.build_index()

            if args.cluster_dstore:
                print('building clusters')
                knn_wrapper.cluster_dstore(num_clusters=args.num_clusters, sample_size=args.sample_size, model=model)

            if args.use_pattern:
                print('using pattern store to enchance patch generation')
                test_acc=test_acc_epoch(args, test_dataloader, model, tokenizer,graphs_by_id=graphs_by_id if graphs_by_id else None)
                result_str = " em: %.4f" % (test_acc)
                logger.info(result_str)
                print(f'{args.k} {args.lmbda} {args.knn_temp}: {result_str}')
                fa.write(result_str)
                logger.info("Finish and take {}".format(get_elapse_time(t0)))
                fa.write("Finish and take {}".format(get_elapse_time(t0)))
                fa.close()


            if knn_wrapper is not None:
                knn_wrapper.break_out()

            
            exit()

        # state_dict = adjust_state_dict_keys(state_dict)

        test_acc=test_acc_epoch(args, test_dataloader, model, tokenizer,graphs_by_id=graphs_by_id if graphs_by_id else None)

        result_str = " em: %.4f" % (test_acc)
        print(f'{args.k} {args.lmbda} {args.knn_temp}: {result_str}')
        logger.info(result_str)
        fa.write(result_str)
        # if args.res_fn:
        #     with open(args.res_fn, 'a+') as f:
        #         f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
        #         f.write(result_str)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()




if __name__ == "__main__":
    main()
