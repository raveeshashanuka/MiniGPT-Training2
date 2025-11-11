import torch
import torch.nn as nn
import yaml

with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Transformer-based LLM from scratch (GPT-like decoder-only)
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, mask=None):
        B, T = x.shape
        x = self.embed(x) + self.pos_embed[:, :T, :]
        memory = torch.zeros_like(x)  # Dummy for decoder
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=mask)
        x = self.norm(x)
        return self.head(x)
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -CONFIG['max_seq_len']:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return idx