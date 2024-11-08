import sys

import itertools
from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
import torch
from torch import nn
from DDFA.code_gnn.models.base_module import BaseModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import logging

logger = logging.getLogger(__name__)

allfeats = [
    "api", "datatype", "literal", "operator"
]

@MODEL_REGISTRY
class FlowGNNGGNNModule(BaseModule):
    def __init__(self,
                 feat,
                 input_dim,
                 hidden_dim,
                 n_steps,
                 num_output_layers,
                 label_style="graph",
                 concat_all_absdf=False,
                 encoder_mode=False,
                 code_embedding_dim=768,  # CodeT5 embedding dimension
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if "_ABS_DATAFLOW" in feat:
            feat = "_ABS_DATAFLOW"
        self.feature_keys = {
            "feature": feat,
        }

        self.input_dim = input_dim
        self.concat_all_absdf = concat_all_absdf

        embedding_dim = hidden_dim
        if self.concat_all_absdf:
            self.all_embeddings = nn.ModuleDict({
                of: nn.Embedding(input_dim, embedding_dim) for of in allfeats
            })
            embedding_dim *= len(allfeats)
            hidden_dim *= len(allfeats)
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)


        self.ggnn = GatedGraphConv(in_feats=embedding_dim,
                                   out_feats=hidden_dim,
                                   n_steps=n_steps,
                                   n_etypes=1)


        self.code_embedding_dim = code_embedding_dim


        self.token_aggregation = nn.GRU(code_embedding_dim, hidden_dim, batch_first=True)


        output_in_size = hidden_dim + embedding_dim  # GGNN output + node features
        self.out_dim = output_in_size


    def forward(self, graph, extrafeats):
        """
        graph: DGL graph object
        extrafeats: extra features (node-related features)
        code_embeddings: precomputed embeddings from CodeT5 for the 'code' in nodes
        """
        code_embeddings = graph.ndata['feat']  

        if self.concat_all_absdf:
            cfeats = []
            for otherfeat in allfeats:
                feat = graph.ndata[f"_ABS_DATAFLOW_{otherfeat}"]
                cfeats.append(self.all_embeddings[otherfeat](feat))
            feat_embed = torch.cat(cfeats, dim=1)  
        else:
            feat = graph.ndata[self.feature_keys["feature"]]
            feat_embed = self.embedding(feat)


        ggnn_out = self.ggnn(graph, feat_embed)

        code_embeddings = code_embeddings.view(-1, code_embeddings.size(1), self.code_embedding_dim)  # [node_num, seq_len, hid_dim]


        output, code_embeddings_agg = self.token_aggregation(code_embeddings)


        code_embeddings = code_embeddings_agg.squeeze(0)
        # print('code_embeddings',code_embeddings.size())


        out = torch.cat([ggnn_out, feat_embed, code_embeddings], dim=-1)

        logits = out  # Remove pooling if not needed

        return logits