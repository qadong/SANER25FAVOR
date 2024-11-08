import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
import sys

from modeling_t5 import T5ForConditionalGeneration

class FLOWT5(T5ForConditionalGeneration):
    def __init__(self, config,flow_gnn):
        super().__init__(config)
        self.flow_gnn = flow_gnn
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FLOWT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, graph=None, **kwargs):

        ggnn_output = None
        padding_length = 200
        if graph is not None:
            ggnn_output = torch.zeros(input_ids.size(0), padding_length, 768)
            num_nodes = []
            i = 0
            for g in graph:
                
                out = self.flow_gnn(g, {})  # ggnn_output 的形状为 (batch_size,seq, 512)
                if out.size(0)<padding_length:
                    ggnn_output[i, :out.size(0)] = out 
                else:
                    ggnn_output[i, :, :] = out[:padding_length,:] 
                num_nodes.append(out.size(0))
                i = i + 1
        self.encoder.gnn_out = ggnn_output

        # print(ggnn_output.size())

        if input_ids is not None:

            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1).long()  # 确保 input_ids 是 LongTensor
            if ggnn_output is not None:
                padding = torch.zeros(input_ids.size(0), padding_length, dtype=input_ids.dtype, device=input_ids.device)  # (batch_size, num_nodes)


    
                input_ids = torch.cat((input_ids, padding), dim=1)

        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)  # 确保 attention_mask 也是 2D
            if ggnn_output is not None:

                padding = torch.ones(input_ids.size(0), padding_length, dtype=input_ids.dtype, device=input_ids.device)  # (batch_size, num_nodes)
                for i, num in enumerate(num_nodes):

                    padding[i, :num] = 1

                attention_mask = torch.cat((attention_mask, padding), dim=1)

        # print('input_ids',input_ids.size())
        # print('attention_mask',attention_mask.size())
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            **kwargs,
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, graph,**kwargs):
        self.encoder.n_passages = input_ids.size(1)
        ggnn_output = None
        padding_length = 200
        if graph is not None:
            ggnn_output = torch.zeros(input_ids.size(0), padding_length, 768)
            num_nodes = []
            i = 0
            for g in graph:
                
                out = self.flow_gnn(g, {})  
                if out.size(0)<padding_length:
                    ggnn_output[i, :out.size(0)] = out 
                else:
                    ggnn_output[i, :, :] = out[:padding_length,:] 
                num_nodes.append(out.size(0))
                i = i + 1
        self.encoder.gnn_out = ggnn_output

        if input_ids is not None:

            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1).long()  # 确保 input_ids 是 LongTensor
            if ggnn_output is not None:
                padding = torch.zeros(input_ids.size(0), padding_length, dtype=input_ids.dtype, device=input_ids.device)  # (batch_size, num_nodes)

    
                input_ids = torch.cat((input_ids, padding), dim=1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)  # 确保 attention_mask 也是 2D
            if ggnn_output is not None:

                padding = torch.ones(input_ids.size(0), padding_length, dtype=input_ids.dtype, device=input_ids.device)  # (batch_size, num_nodes)
                for i, num in enumerate(num_nodes):

                    padding[i, :num] = 1

                attention_mask = torch.cat((attention_mask, padding), dim=1)

        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict,strict=True)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.main_input_name = "input_ids"

        self.encoder = encoder
        self.gnn_out = None
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)


    def forward(self, input_ids=None, attention_mask=None, ggnn_output=None, **kwargs):
        # print(input_ids.size(),attention_mask.size())
        outputs = self.encoder(input_ids[:,:512], attention_mask[:,:512], **kwargs)
        # print('input_ids', input_ids.size())
        if self.gnn_out is not None:
            self.gnn_out = self.gnn_out.to(input_ids.device)
            encoder_hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            # print(encoder_hidden_states.size())


            encoder_hidden_states = torch.cat((encoder_hidden_states, self.gnn_out), dim=1)  # 替换填充部分
            # print(encoder_hidden_states.size())


            # print('outputs.last_hidden_state',outputs.last_hidden_state)
            outputs.last_hidden_state = encoder_hidden_states

        # print(outputs)
        return outputs
class CheckpointWrapper(torch.nn.Module):
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output




# import torch
# from transformers import AutoTokenizer
# import transformers
# import dgl
# tokenizer = AutoTokenizer.from_pretrained(
#     '/home/hdd/qingao/cache/huggingface/transformers/models--Salesforce--codet5-base/snapshots/4078456db09ba972a3532827a0b5df4da172323c'
#     )
# tokenizer.add_tokens(["Vul_Start","Vul_End"])
# model = transformers.T5ForConditionalGeneration.from_pretrained(
#     '/home/hdd/qingao/cache/huggingface/transformers/models--Salesforce--codet5-base/snapshots/4078456db09ba972a3532827a0b5df4da172323c'
#     )
# model.resize_token_embeddings(len(tokenizer))
# model.load_state_dict(torch.load('/home/hdd/qingao/DeepDFA/CodeT5/saved_models/repair/codeT5/checkpoint-best-acc/pytorch_model.bin'))
# config = model.config


# input_dim = 8
# feat = "_ABS_DATAFLOW_datatype_all_limitall_1000_limitsubkeys_1000"
# gtype = "cfg"
# label_style = "graph"
# dsname = "bigvul"
# node_type_feat = None
# concat_all_absdf = True
# hidden_dim = 64
# n_steps = 5
# num_output_layers = 3

# flowgnn_model = FlowGNNGGNNModule(
#     feat,
#     input_dim,
#     hidden_dim,
#     n_steps,
#     num_output_layers,
#     label_style=label_style,
#     # freeze_graph=False,
#     # append_dataflow="before_graph",
#     # codebert_feat=None,
#     # doc2vec_feat=None,
#     # glove_feat=None,
#     # num_node_types=flowgnn_datamodule.num_node_types,
#     # node_type_feat=node_type_feat,
#     # just_codebert=False,
#     concat_all_absdf=concat_all_absdf,
#     # undersample_node_on_loss_factor=None,
#     # test_every=False,
#     # tune_nni=False,
#     # positive_weight=None,
#     encoder_mode=True,
# )


# flow_model = FLOWT5(config,flow_gnn=flowgnn_model)
# flow_model.load_t5(model.state_dict())