import pytest
import torch
from scripts.train import TextDataset

@pytest.fixture
def sample_tokens():
    return list(range(100))

def test_dataset(sample_tokens):
    dataset = TextDataset(sample_tokens, seq_len=10)
    assert len(dataset) == 90
    src, tgt = dataset[0]
    assert torch.equal(src, torch.tensor(sample_tokens[0:10]))
    assert torch.equal(tgt, torch.tensor(sample_tokens[1:11]))