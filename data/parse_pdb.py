# data/parse_pdb.py
"""
Parse PDB files to extract antibody chain coordinates, residue types,
and paratope labels (residues in contact with antigen).
"""

import numpy as np
from pathlib import Path
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa

# Standard 20 amino acids + unknown
AA_VOCAB = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20
}

# Physicochemical properties: [polarity, charge]
# polarity: 0=nonpolar, 1=polar; charge: -1/0/1
AA_PROPERTIES = {
    'ALA': [0, 0], 'ARG': [1, 1], 'ASN': [1, 0], 'ASP': [1, -1], 'CYS': [1, 0],
    'GLN': [1, 0], 'GLU': [1, -1], 'GLY': [0, 0], 'HIS': [1, 1], 'ILE': [0, 0],
    'LEU': [0, 0], 'LYS': [1, 1], 'MET': [0, 0], 'PHE': [0, 0], 'PRO': [0, 0],
    'SER': [1, 0], 'THR': [1, 0], 'TRP': [1, 0], 'TYR': [1, 0], 'VAL': [0, 0],
    'UNK': [0, 0]
}

ANTIBODY_CHAINS = {'H', 'L'}  # Heavy and Light chains by SAbDab convention
CONTACT_THRESHOLD = 4.5  # Angstroms for antigen contact


def get_ca_atoms(chain):
    """Extract CA atoms from a chain, returning residue name, CA coords."""
    residues = []
    for res in chain:
        if not is_aa(res, standard=True):
            continue
        if 'CA' not in res:
            continue
        resname = res.get_resname()
        ca_coord = np.array(res['CA'].get_vector().get_array())
        residues.append({
            'resname': resname,
            'coord': ca_coord,
            'res_id': res.get_id()
        })
    return residues


def compute_paratope_labels(ab_residues, ag_residues, threshold=CONTACT_THRESHOLD):
    """
    Label antibody residues as paratope (1) if any CA atom of antigen
    is within `threshold` Angstroms. Uses CA-CA distance as proxy.
    """
    labels = []
    ag_coords = np.array([r['coord'] for r in ag_residues])

    for ab_res in ab_residues:
        ab_coord = ab_res['coord'].reshape(1, 3)
        if len(ag_coords) == 0:
            labels.append(0)
            continue
        dists = np.linalg.norm(ag_coords - ab_coord, axis=1)
        labels.append(1 if dists.min() <= threshold else 0)

    return labels


def parse_pdb(pdb_path: str):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('ab', pdb_path)
    except Exception as e:
        print(f"Failed to parse {pdb_path}: {e}")
        return None

    model = structure[0]
    ab_residues = []
    ag_residues = []

    chains = list(model.get_chains())
    chain_ids = [c.get_id().upper() for c in chains]

    # Check if standard SAbDab H/L chains exist
    has_hl = any(cid in ANTIBODY_CHAINS for cid in chain_ids)

    for chain in chains:
        chain_id = chain.get_id().upper()
        residues = get_ca_atoms(chain)
        if not residues:
            continue
        if has_hl:
            if chain_id in ANTIBODY_CHAINS:
                ab_residues.extend(residues)
            else:
                ag_residues.extend(residues)
        else:
            # Fallback: largest 2 chains = antibody, rest = antigen
            ab_residues.extend(residues)  # collect all first

    # Fallback: if no antigen found, split by chain count
    if len(ag_residues) == 0 and not has_hl:
        # Sort chains by size, treat smallest as antigen
        chain_data = []
        for chain in chains:
            res = get_ca_atoms(chain)
            if res:
                chain_data.append(res)
        chain_data.sort(key=len, reverse=True)
        # Treat top 2 chains as antibody, rest as antigen
        ab_residues = []
        for i, res in enumerate(chain_data):
            if i < 2:
                ab_residues.extend(res)
            else:
                ag_residues.extend(res)

    if len(ab_residues) == 0:
        return None

    labels = compute_paratope_labels(ab_residues, ag_residues)
    coords = np.array([r['coord'] for r in ab_residues], dtype=np.float32)
    features = []
    for r in ab_residues:
        rn = r['resname'] if r['resname'] in AA_VOCAB else 'UNK'
        aa_idx = AA_VOCAB[rn]
        props = AA_PROPERTIES[rn]
        features.append([aa_idx] + props)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.long)
    residue_names = [r['resname'] for r in ab_residues]

    return {
        'coords': coords,
        'features': features,
        'labels': labels,
        'residue_names': residue_names,
        'pdb_path': str(pdb_path)
    }