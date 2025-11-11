import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import os
import logging
import yaml

with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Assuming model.py, data_prep.py have run; import from them
# For standalone, include necessary parts

# Config updates for distributed
CONFIG['world_size'] = 1  # Default; set by torchrun
CONFIG['rank'] = 0
CONFIG['local_rank'] = int(os.environ.get('LOCAL_RANK', 0))  # For multi-GPU

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)  # NCCL for GPU

def cleanup():
    dist.destroy_process_group()

class TextDataset(Dataset):  # Same as before
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx:idx+self.seq_len]), torch.tensor(self.tokens[idx+1:idx+self.seq_len+1])

class MiniGPT(nn.Module):  # Same as before
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
        memory = torch.zeros_like(x)  # Dummy
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=mask)
        x = self.norm(x)
        return self.head(x)

def train(rank, world_size):
    setup(rank, world_size)
    CONFIG['device'] = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None
    
    logger = logging.getLogger(__name__)
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)  # Reduce logging on non-master
    
    # Load data (assume prepared)
    tokens = torch.load(os.path.join(CONFIG['data_dir'], 'tokens.pt'))
    dataset = TextDataset(tokens, CONFIG['max_seq_len'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, sampler=sampler)
    
    model = MiniGPT(CONFIG['vocab_size'], CONFIG['embed_dim'], CONFIG['num_heads'], CONFIG['num_layers'], CONFIG['max_seq_len']).to(CONFIG['device'])
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if CONFIG['mixed_precision'] else None
    
    if rank == 0:
        logger.info(f"Distributed training on {world_size} processes")
    
    for epoch in range(CONFIG['epochs']):
        sampler.set_epoch(epoch)  # For shuffling
        total_loss = torch.zeros(1).to(CONFIG['device'])
        for src, tgt in dataloader:
            src, tgt = src.to(CONFIG['device']), tgt.to(CONFIG['device'])
            mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(CONFIG['device'])
            
            optimizer.zero_grad()
            if CONFIG['mixed_precision']:
                with autocast():
                    out = model(src, mask)
                    loss = criterion(out.view(-1, CONFIG['vocab_size']), tgt.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(src, mask)
                loss = criterion(out.view(-1, CONFIG['vocab_size']), tgt.view(-1))
                loss.backward()
                optimizer.step()
            
            total_loss += loss.detach()
        
        # All reduce loss for average
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        avg_loss = total_loss.item() / (len(dataloader) * world_size)
        
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']}, Avg Loss: {avg_loss:.4f}")
    
    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(CONFIG['model_dir'], 'minigpt_distributed.pth'))
        logger.info("Distributed model trained and saved (from rank 0).")
    
    cleanup()

# For DeepSpeed integration (optional; uncomment and install deepspeed)
# import deepspeed
# def train_ds(rank, world_size):
#     setup(rank, world_size)
#     model = MiniGPT(...)  # As above
#     model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
#                                                          config={'train_batch_size': CONFIG['batch_size'] * world_size,
#                                                                  'fp16': {'enabled': True}})
#     # Then use model_engine for forward/backward/step
#     # Adjust dataloader accordingly

if __name__ == '__main__':
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    CONFIG['world_size'] = world_size
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)