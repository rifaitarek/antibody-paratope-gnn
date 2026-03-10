"""
Convert parsed PDB data into PyTorch Geometric Data objects (graphs).
Nodes = amino acids. Edges = spatial neighbors within distance cutoff.
"""

import torch
import numpy as np
from torch_geometric.data import Data


def build_graph(parsed_data: dict, distance_cutoff: float = 10.0) -> Data:
    """
    Build a PyG Data object from parsed PDB data.

    Args:
        parsed_data: dict with keys 'coords', 'features', 'labels'
        distance_cutoff: max CA-CA distance (Angstroms) to form an edge

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr, y, pos
    """
    coords = parsed_data['coords']   # (N, 3)
    features = parsed_data['features']  # (N, 3)
    labels = parsed_data['labels']   # (N,)

    N = len(coords)

    # Build edge index via distance matrix
    coord_tensor = torch.tensor(coords, dtype=torch.float)
    diff = coord_tensor.unsqueeze(0) - coord_tensor.unsqueeze(1)  # (N, N, 3)
    dist_matrix = torch.norm(diff, dim=-1)  # (N, N)

    # Edges where distance < cutoff, excluding self-loops
    mask = (dist_matrix < distance_cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # (2, E)

    # Edge attributes: inverse distance
    edge_dist = dist_matrix[edge_index[0], edge_index[1]].unsqueeze(1)
    edge_attr = 1.0 / (edge_dist + 1e-6)

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=coord_tensor,
        num_nodes=N
    )
    return data