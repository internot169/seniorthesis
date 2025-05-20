import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoTokenizer

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

class Graphormer(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.6, alpha=0.2, nheads=1):
        super(Graphormer, self).__init__()
        self.dropout = dropout
        
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(nout, nout * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nout * 4, nout)
        )
        
        self.layer_norm1 = nn.LayerNorm(nout)
        self.layer_norm2 = nn.LayerNorm(nout)
        
        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.attentions:
            attention.reset_parameters()
        self.out_att.reset_parameters()

    def forward(self, x, adj):
        residual = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        x = self.layer_norm1(residual + x)
        
        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm2(residual + x)
        
        return x

    def compute_loss(self, pred_attention, target_attention):
        return -torch.mean(torch.sum(target_attention * torch.log(pred_attention + 1e-10), dim=1))

def create_knowledge_graph(attention_matrix, threshold):
    adj_matrix = (attention_matrix > threshold).float()
    edge_weights = attention_matrix * adj_matrix
    return adj_matrix, edge_weights

def deepseek():
    model_name = "deepseek-ai/deepseek-coder-1.5b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    hidden_size = base_model.config.hidden_size
    num_heads = base_model.config.num_attention_heads
    
    graphormer = Graphormer(
        nfeat=hidden_size,
        nhid=hidden_size // num_heads,
        nout=hidden_size,
        nheads=num_heads
    )
    
    with torch.no_grad():
        for i, attention in enumerate(graphormer.attentions):
            attention.W.copy_(base_model.encoder.layer[i].attention.self.query.weight[:hidden_size//num_heads].T)
            attention.a.copy_(base_model.encoder.layer[i].attention.self.key.weight[:hidden_size//num_heads].T)
        
        graphormer.feed_forward[0].weight.copy_(base_model.encoder.layer[0].intermediate.dense.weight)
        graphormer.feed_forward[0].bias.copy_(base_model.encoder.layer[0].intermediate.dense.bias)
        graphormer.feed_forward[3].weight.copy_(base_model.encoder.layer[0].output.dense.weight)
        graphormer.feed_forward[3].bias.copy_(base_model.encoder.layer[0].output.dense.bias)
    
    return graphormer, tokenizer

graphormer, tokenizer = deepseek()