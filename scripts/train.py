import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
from src.model import MiniGPT
import yaml

with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Set up logging for monitoring (non-functional: observability)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx:idx+self.seq_len]), torch.tensor(self.tokens[idx+1:idx+self.seq_len+1])

def train_model():
    tokens = torch.load(os.path.join(CONFIG['data_dir'], 'tokens.pt'))
    dataset = TextDataset(tokens, CONFIG['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = MiniGPT(CONFIG['vocab_size'], CONFIG['embed_dim'], CONFIG['num_heads'], CONFIG['num_layers'], CONFIG['max_seq_len']).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if CONFIG['mixed_precision'] else None
    
    logger.info(f"Training on {CONFIG['device']}")
    
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
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
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), os.path.join(CONFIG['model_dir'], 'minigpt.pth'))
    logger.info("Model trained and saved.")

if __name__ == "__main__":  
    train_model()