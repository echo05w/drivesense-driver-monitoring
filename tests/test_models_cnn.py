"""Shape/smoke tests for CNN architectures — verifies the modules are wired
correctly, not that they classify anything (no training happens here).

TransferLearningCNN is excluded from the default test run: it downloads
pretrained ImageNet weights on first construction, which is legitimate but
too slow/network-dependent for a fast local test suite. It was manually
verified working (see docs/Master_Plan_Status.md) and is exercised for real
once training actually begins.
"""

import torch

from drivesense.models.cnn import SimpleCNN


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out.shape == (4, 10)


def test_simple_cnn_handles_different_input_sizes():
    model = SimpleCNN(num_classes=5)
    for size in (32, 96, 128):
        x = torch.randn(1, 3, size, size)
        out = model(x)
        assert out.shape == (1, 5)


def test_simple_cnn_is_trainable_single_step():
    model = SimpleCNN(num_classes=3)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
