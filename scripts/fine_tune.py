import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from src.model import MiniGPT
import yaml

with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Set up logging for monitoring (non-functional: observability)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fine-tuning for a task: Add classification head for sentiment, then fine-tune
class FineTuneDataset(Dataset):
    def __init__(self, tokens_list, labels, seq_len):
        self.tokens_list = tokens_list
        self.labels = labels
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.tokens_list)
    
    def __getitem__(self, idx):
        tokens = self.tokens_list[idx][:self.seq_len]
        padded = tokens + [0] * (self.seq_len - len(tokens))  # Pad with 0 (assume 0 is padding)
        return torch.tensor(padded), torch.tensor(self.labels[idx])

class MiniGPTForClassification(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(CONFIG['embed_dim'], 2)  # Binary sentiment
    
    def forward(self, x, mask=None):
        out = self.base(x, mask)[:, -1, :]  # Last token embedding? Wait, actually pool
        pooled = torch.mean(out, dim=1)  # Mean pool for classification
        return self.classifier(pooled)

def fine_tune_model():
    # Load pre-trained
    base_model = MiniGPT(CONFIG['vocab_size'], CONFIG['embed_dim'], CONFIG['num_heads'], CONFIG['num_layers'], CONFIG['max_seq_len'])
    base_model.load_state_dict(torch.load(os.path.join(CONFIG['model_dir'], 'minigpt.pth')))
    model = MiniGPTForClassification(base_model).to(CONFIG['device'])
    
    tokens_list = torch.load(os.path.join(CONFIG['data_dir'], 'finetune_tokens.pt'))
    labels = torch.load(os.path.join(CONFIG['data_dir'], 'finetune_labels.pt'))
    dataset = FineTuneDataset(tokens_list, labels, CONFIG['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'] / 10)  # Lower LR for fine-tune
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if CONFIG['mixed_precision'] else None
    
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(CONFIG['device']), tgt.to(CONFIG['device'])
            mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(CONFIG['device'])
            
            optimizer.zero_grad()
            if CONFIG['mixed_precision']:
                with autocast():
                    out = model(src, mask)
                    loss = criterion(out, tgt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(src, mask)
                loss = criterion(out, tgt)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Fine-tune Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), os.path.join(CONFIG['model_dir'], 'minigpt_finetuned.pth'))
    logger.info("Model fine-tuned and saved.")

if __name__ == "__main__":  # For fine_tune.py
    fine_tune_model()