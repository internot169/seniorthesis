import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(2*out_dim, 1))
        
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
        Wh1 = torch.matmul(Wh, self.a[:self.out_dim, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_dim:, :])
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.6, alpha=0.2, nheads=1):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        self.attentions = [GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        self.out_att = GATLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

def train_gat(model, optimizer, features, adj, labels, epochs=200):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}: loss = {loss.item():.4f}')

num_nodes = 1000
input_dim = 64
hidden_dim = 128
output_dim = 32

batch_size = 32
node_indices = torch.randint(0, num_nodes, (batch_size,))

adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
adj_matrix = adj_matrix / (adj_matrix.sum(dim=1, keepdim=True) + 1e-8)

model = NodeEmbedding(num_nodes, input_dim, hidden_dim, output_dim)

embeddings = model(node_indices, adj_matrix)