import pytest
import torch
from src.model import MiniGPT

@pytest.fixture
def config():
    return {
        'vocab_size': 100,
        'embed_dim': 32,
        'num_heads': 2,
        'num_layers': 1,
        'max_seq_len': 10
    }

def test_model_forward(config):
    model = MiniGPT(**config)
    input = torch.randint(0, config['vocab_size'], (2, 5))
    output = model(input)
    assert output.shape == (2, 5, config['vocab_size'])

def test_model_generate(config):
    model = MiniGPT(**config)
    input = torch.tensor([[1]])
    generated = model.generate(input, max_new_tokens=5)
    assert generated.shape == (1, 6)  # Original + 5 new