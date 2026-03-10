# AbParatope-GNN

> Graph Neural Network for Antibody Paratope Prediction from 3D Structure

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-yellow.svg)](https://wandb.ai/)

## Overview

AbParatope-GNN predicts which amino acids on an antibody form the **paratope** — the binding site that contacts an antigen — directly from 3D PDB structure files. Antibody structures are represented as graphs (nodes = residues, edges = spatial neighbors) and processed with a Graph Convolutional Network.

## Architecture
```
PDB File → CA Extraction → Distance Graph → GCN → Node Labels (paratope/non-paratope)
```

- **Nodes**: amino acids with features [AA type, polarity, charge]  
- **Edges**: residue pairs within 10Å (configurable)  
- **Model**: Multi-layer GCN with batch norm, residual connections, weighted cross-entropy

## Installation
```bash
git clone https://github.com/rifaitarek/AbParatope-GNN.git
cd AbParatope-GNN
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install -r requirements.txt
```

## Usage

**1. Configure W&B (first time):**
```bash
wandb login
```

**2. Run training with defaults:**
```bash
python train.py
```

**3. Override hyperparameters via CLI (Hydra):**
```bash
python train.py model.num_layers=4 optim.lr=0.0005
python train.py model.hidden_channels=128 data.distance_cutoff=8.0
python train.py optim.epochs=100 optim.batch_size=16
```

## Running Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

## Project Structure
```
AbParatope-GNN/
├── conf/config.yaml        # Hydra config (hyperparameters)
├── data/
│   ├── parse_pdb.py        # Biopython PDB parser
│   ├── graph_builder.py    # PyG graph construction
│   └── dataset.py          # PyG Dataset class
├── models/
│   └── gcn.py              # GCN architecture
├── utils/
│   ├── metrics.py          # F1, AUC, precision/recall
│   └── helpers.py          # Seeding, splitting, download
├── tests/
│   ├── test_graph.py       # Graph pipeline unit tests
│   └── test_model.py       # Model shape/training tests
├── train.py                # Main training script
└── requirements.txt
```

## Key Design Choices

| Decision | Rationale |
|---|---|
| CA-only graph | Efficient; CA captures backbone geometry well |
| Weighted cross-entropy | Paratope residues are ~10% of all residues (class imbalance) |
| Residual GCN | Avoids oversmoothing in deeper networks |
| F1 as primary metric | More informative than accuracy for imbalanced labels |