# models/gcn.py
"""
Graph Convolutional Network for paratope prediction (binary node classification).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class ParatopeGCN(nn.Module):
    """
    Multi-layer GCN with skip connections for paratope prediction.
    
    Message passing:
        h_i^(l+1) = sigma( sum_{j in N(i)} (1/c_ij) * W^(l) * h_j^(l) )
    """

    def __init__(self, node_features: int, hidden_channels: int,
                 num_layers: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_channels)

        # GCN layers
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # Output head: binary classification per node
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )

    def forward(self, x, edge_index, batch=None):
        # Project input features
        x = F.relu(self.input_proj(x))

        # Message passing layers with residual connections
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual

        # Node-level classification
        out = self.classifier(x)
        return out  # logits: (N, 2)