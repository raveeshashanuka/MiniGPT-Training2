import torch

# Simple char tokenizer (from scratch)
class CharTokenizer:
    def __init__(self, vocab_path):
        self.vocab = torch.load(vocab_path)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        return [self.vocab[ch] for ch in text if ch in self.vocab]
    
    def decode(self, tokens):
        return ''.join([self.inv_vocab[t] for t in tokens])