"""Shape/smoke tests for temporal drowsiness models. Verifies the modules
are wired correctly on synthetic tensors — no training/evaluation claims."""

import pytest
import torch

from drivesense.models.temporal import DrowsinessGRU, DrowsinessTemporalCNN


@pytest.mark.parametrize("bidirectional", [False, True])
def test_gru_output_shape(bidirectional):
    model = DrowsinessGRU(input_dim=5, num_classes=3, hidden_dim=16, num_layers=2, bidirectional=bidirectional)
    x = torch.randn(4, 30, 5)
    out = model(x)
    assert out.shape == (4, 3)


def test_temporal_cnn_output_shape():
    model = DrowsinessTemporalCNN(input_dim=5, num_classes=3)
    x = torch.randn(4, 30, 5)
    out = model(x)
    assert out.shape == (4, 3)


def test_temporal_cnn_handles_variable_seq_len():
    model = DrowsinessTemporalCNN(input_dim=5, num_classes=3)
    for seq_len in (10, 30, 60):
        x = torch.randn(2, seq_len, 5)
        out = model(x)
        assert out.shape == (2, 3)


@pytest.mark.parametrize("model_cls,kwargs", [
    (DrowsinessGRU, {"input_dim": 5, "num_classes": 3, "hidden_dim": 16}),
    (DrowsinessTemporalCNN, {"input_dim": 5, "num_classes": 3}),
])
def test_single_training_step_produces_finite_loss(model_cls, kwargs):
    model = model_cls(**kwargs)
    x = torch.randn(4, 30, 5)
    y = torch.tensor([0, 1, 2, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
