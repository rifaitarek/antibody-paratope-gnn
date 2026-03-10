"""
PyTorch Geometric Dataset that wraps our PDB parsing + graph building pipeline.
"""

import os
import torch
from pathlib import Path
from torch_geometric.data import Dataset
from data.parse_pdb import parse_pdb
from data.graph_builder import build_graph


class ParatopeDataset(Dataset):
    def __init__(self, pdb_dir: str, processed_dir: str, distance_cutoff: float = 10.0,
                 transform=None, pre_transform=None):
        self.pdb_dir = Path(pdb_dir)
        self.distance_cutoff = distance_cutoff
        self.processed_dir_path = Path(processed_dir)
        self.processed_dir_path.mkdir(parents=True, exist_ok=True)
        self._pdb_files = sorted(self.pdb_dir.glob("*.pdb"))
        self._processed_files = []
        self._preprocess()
        super().__init__(str(processed_dir), transform, pre_transform)

    def _preprocess(self):
        for pdb_file in self._pdb_files:
            out_path = self.processed_dir_path / (pdb_file.stem + ".pt")
            if not out_path.exists():
                parsed = parse_pdb(str(pdb_file))
                if parsed is None:
                    continue
                graph = build_graph(parsed, self.distance_cutoff)
                torch.save(graph, str(out_path))
            if out_path.exists():
                self._processed_files.append(out_path)

    def len(self):
        return len(self._processed_files)

    def get(self, idx):
        return torch.load(str(self._processed_files[idx]), weights_only=False)