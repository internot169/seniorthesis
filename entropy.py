import torch
import torch.nn as nn
import numpy as np
from graphormer import Graphormer
import math

def get_weight_entropy(model):
    total = 0
    params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.detach().cpu().numpy()
            hist, bins = np.histogram(weights, bins=50, density=True)
            probs = hist * np.diff(bins)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            total += entropy * param.numel()
            params += param.numel()
    
    return total / params if params > 0 else 0

def get_attention_entropy(attention):
    probs = F.softmax(attention, dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
    return entropy.item()

def get_graph_entropy(adj, weights):
    degrees = adj.sum(dim=1)
    edges = degrees.sum()
    if edges == 0:
        return 0
    
    probs = degrees / edges
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
    return entropy.item()

def get_mdl(model, x, adj):
    with torch.no_grad():
        out = model(x, adj)
        weight_ent = get_weight_entropy(model)
        attn_ent = get_attention_entropy(out)
        adj_mat, edge_w = model.create_knowledge_graph(out, threshold=0.1)
        graph_ent = get_graph_entropy(adj_mat, edge_w)
        total = weight_ent + attn_ent + graph_ent
        
        return {
            'weight': weight_ent,
            'attention': attn_ent,
            'graph': graph_ent,
            'total': total
        }

def main():
    model = Graphormer(
        nfeat=512,
        nhid=256,
        nout=512,
        nheads=8
    )
    
    batch = 32
    seq = 100
    x = torch.randn(batch, seq, 512)
    adj = torch.ones(batch, seq, seq)
    
    metrics = get_mdl(model, x, adj)
    
    print(f"Weights: {metrics['weight']:.4f}")
    print(f"Attention: {metrics['attention']:.4f}")
    print(f"Graph: {metrics['graph']:.4f}")
    print(f"Total: {metrics['total']:.4f}")

if __name__ == "__main__":
    main()
