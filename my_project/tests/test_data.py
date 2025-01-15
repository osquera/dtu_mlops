import os

import pytest
import torch
from torch.utils.data import Dataset

from my_project.data import MyDataset, corrupt_mnist

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Train dataset did not have the correct number of samples"
    assert len(test) == 5000, "Test dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all(), "Train targets were not correct"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all(), "Test targets were not correct"
