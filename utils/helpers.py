"""Utility functions: seeding, data splitting, download helpers."""

import os
import random
import torch
import numpy as np
import urllib.request
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, train_fraction: float = 0.8):
    """Randomly split a dataset into train/val subsets."""
    n = len(dataset)
    n_train = int(n * train_fraction)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return (
        [dataset[i] for i in train_idx],
        [dataset[i] for i in val_idx]
    )


def download_sabdab_sample(raw_dir: str, n_structures: int = 20):
    """
    Download a small sample of antibody-antigen complexes from SAbDab.
    Uses the SAbDab summary file to get PDB IDs, then downloads from RCSB.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # SAbDab summary of non-redundant Ab-Ag complexes
    summary_url = (
        "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
        "?ABtype=All&method=All&species=All&antigen_type=All"
        "&has_antigen=True&is_bound=True&resolution=2.5&rfactor=0.3"
        "&prominence=1&chothia_defined=1&format=tsv"
    )

    summary_path = raw_path / "sabdab_summary.tsv"

    if not summary_path.exists():
        print("Downloading SAbDab summary...")
        try:
            urllib.request.urlretrieve(summary_url, str(summary_path))
        except Exception as e:
            print(f"Could not download summary: {e}")
            print("Creating mock PDB files for testing instead...")
            _create_mock_pdbs(raw_path, n_structures)
            return

    # Parse PDB IDs from summary
    pdb_ids = []
    with open(summary_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            parts = line.strip().split('\t')
            if parts:
                pdb_ids.append(parts[0].lower())
            if len(pdb_ids) >= n_structures:
                break

    # Download PDB files from RCSB
    downloaded = 0
    for pdb_id in pdb_ids:
        out_path = raw_path / f"{pdb_id}.pdb"
        if out_path.exists():
            downloaded += 1
            continue
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, str(out_path))
            print(f"Downloaded {pdb_id}.pdb")
            downloaded += 1
        except Exception as e:
            print(f"Skipped {pdb_id}: {e}")

    print(f"Downloaded {downloaded} PDB files to {raw_path}")


def _create_mock_pdbs(raw_path: Path, n: int):
    """Create minimal mock PDB files for offline testing."""
    import random

    aa_list = ['ALA', 'ARG', 'ASN', 'ASP', 'GLY', 'LEU', 'LYS', 'PHE', 'SER', 'VAL']

    for i in range(n):
        lines = []
        # Heavy chain (H) residues
        for j in range(50):
            x, y, z = random.uniform(0, 50), random.uniform(0, 50), random.uniform(0, 50)
            aa = random.choice(aa_list)
            lines.append(
                f"ATOM  {j+1:5d}  CA  {aa} H{j+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
        # Antigen chain (A) residues — some near H chain to create paratope labels
        for j in range(30):
            x, y, z = random.uniform(0, 60), random.uniform(0, 60), random.uniform(0, 60)
            aa = random.choice(aa_list)
            lines.append(
                f"ATOM  {j+51:5d}  CA  {aa} A{j+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
        lines.append("END")
        pdb_path = raw_path / f"mock_{i:03d}.pdb"
        pdb_path.write_text("\n".join(lines))
    print(f"Created {n} mock PDB files in {raw_path}")