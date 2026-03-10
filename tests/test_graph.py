# tests/test_graph.py
"""Unit tests for graph construction pipeline."""

import pytest
import numpy as np
import torch
from data.parse_pdb import parse_pdb, AA_VOCAB, compute_paratope_labels
from data.graph_builder import build_graph


@pytest.fixture
def mock_pdb(tmp_path):
    """Create a minimal mock PDB file for testing."""
    pdb_content = """ATOM      1  CA  ALA H   1      10.000  10.000  10.000  1.00 20.00           C
ATOM      2  CA  GLY H   2      13.000  10.000  10.000  1.00 20.00           C
ATOM      3  CA  LYS H   3      16.000  10.000  10.000  1.00 20.00           C
ATOM      4  CA  SER H   4      19.000  10.000  10.000  1.00 20.00           C
ATOM      5  CA  PHE H   5      22.000  10.000  10.000  1.00 20.00           C
ATOM      6  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C
ATOM      7  CA  ARG A   2      30.000  30.000  30.000  1.00 20.00           C
END
"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)
    return str(pdb_file)


def test_parse_pdb_returns_dict(mock_pdb):
    result = parse_pdb(mock_pdb)
    assert result is not None, "parse_pdb should return a dict for valid PDB"
    assert 'coords' in result
    assert 'features' in result
    assert 'labels' in result


def test_parse_pdb_correct_shapes(mock_pdb):
    result = parse_pdb(mock_pdb)
    N = len(result['coords'])
    assert N > 0, "Should have at least one residue"
    assert result['coords'].shape == (N, 3)
    assert result['features'].shape == (N, 3)
    assert result['labels'].shape == (N,)


def test_labels_are_binary(mock_pdb):
    result = parse_pdb(mock_pdb)
    unique = set(result['labels'].tolist())
    assert unique.issubset({0, 1}), f"Labels should be 0 or 1, got {unique}"


def test_build_graph_has_nodes_and_edges(mock_pdb):
    result = parse_pdb(mock_pdb)
    graph = build_graph(result, distance_cutoff=10.0)
    assert graph.num_nodes > 0, "Graph must have nodes"
    assert graph.edge_index.shape[0] == 2, "edge_index must have shape (2, E)"
    assert graph.edge_index.shape[1] > 0, "Graph must have at least one edge"


def test_build_graph_node_features_shape(mock_pdb):
    result = parse_pdb(mock_pdb)
    graph = build_graph(result, distance_cutoff=10.0)
    assert graph.x.shape[0] == graph.num_nodes
    assert graph.x.shape[1] == 3  # aa_index, polarity, charge


def test_distance_cutoff_affects_edges(mock_pdb):
    result = parse_pdb(mock_pdb)
    graph_small = build_graph(result, distance_cutoff=3.0)
    graph_large = build_graph(result, distance_cutoff=20.0)
    assert graph_large.edge_index.shape[1] >= graph_small.edge_index.shape[1], \
        "Larger cutoff should produce >= edges"


def test_paratope_labels_with_close_antigen():
    ab_residues = [
        {'coord': np.array([0.0, 0.0, 0.0])},
        {'coord': np.array([100.0, 100.0, 100.0])},
    ]
    ag_residues = [
        {'coord': np.array([1.0, 0.0, 0.0])},  # very close to ab[0]
    ]
    labels = compute_paratope_labels(ab_residues, ag_residues, threshold=4.5)
    assert labels[0] == 1, "First residue should be paratope (antigen is 1Å away)"
    assert labels[1] == 0, "Second residue should not be paratope (antigen is far)"