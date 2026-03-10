"""Unit tests for GCN model architecture."""

import pytest
import torch
from torch_geometric.data import Data, Batch
from models.gcn import ParatopeGCN


@pytest.fixture
def simple_graph():
    """Create a minimal graph for testing."""
    x = torch.randn(10, 3)  # 10 nodes, 3 features
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
    ], dtype=torch.long)
    y = torch.randint(0, 2, (10,))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=10)


@pytest.fixture
def model():
    return ParatopeGCN(node_features=3, hidden_channels=16, num_layers=2, dropout=0.1)


def test_model_output_shape(model, simple_graph):
    model.eval()
    with torch.no_grad():
        out = model(simple_graph.x, simple_graph.edge_index)
    assert out.shape == (10, 2), f"Expected (10, 2), got {out.shape}"


def test_model_output_is_logits(model, simple_graph):
    """Logits should NOT sum to 1 (unlike softmax)."""
    model.eval()
    with torch.no_grad():
        out = model(simple_graph.x, simple_graph.edge_index)
    row_sums = torch.softmax(out, dim=-1).sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(10), atol=1e-5)


def test_model_trains_without_error(model, simple_graph):
    """One forward + backward pass should not raise errors."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    out = model(simple_graph.x, simple_graph.edge_index)
    loss = torch.nn.functional.cross_entropy(out, simple_graph.y)
    loss.backward()
    optimizer.step()
    assert loss.item() > 0


def test_model_different_layer_counts(simple_graph):
    for n_layers in [1, 2, 4]:
        m = ParatopeGCN(node_features=3, hidden_channels=16, num_layers=n_layers)
        m.eval()
        with torch.no_grad():
            out = m(simple_graph.x, simple_graph.edge_index)
        assert out.shape == (10, 2), f"Failed for num_layers={n_layers}"


def test_model_batch_inference():
    """Test that model works with batched graphs from DataLoader."""
    graphs = []
    for _ in range(4):
        n = torch.randint(5, 15, (1,)).item()
        x = torch.randn(n, 3)
        ei = torch.randint(0, n, (2, n * 2))
        graphs.append(Data(x=x, edge_index=ei, num_nodes=n))

    batch = Batch.from_data_list(graphs)
    model = ParatopeGCN(node_features=3, hidden_channels=16, num_layers=2)
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)
    assert out.shape[1] == 2
    assert out.shape[0] == batch.num_nodes