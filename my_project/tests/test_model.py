import torch
import pytest
from my_project.model import MyAwesomeModel


def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10), f"Expected output shape (1, 10), but got {y.shape}"


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)

# coverage run --omit="tests/*" -m pytest tests/
