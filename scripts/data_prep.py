"""Data cleaning and processsing Pipeline"""

#imports
import os
import torch
import requests
import logging
import string
import yaml

with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Set up logging for monitoring (non-functional: observability)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Download data if the specified file is not in the local machine                         
def download_data(url,filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'w', encoding = 'utf-8') as f:
            f.write(response.text)
        logger.info(f"Downloaded {filename}")

def clean_text(text):
    # Simple cleaning: lowercase, remove punctuation, normalize spaces
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def build_vocab(text, max_vocab=10000):
    # Character-level for simplicity, but could be word/BPE for scale
    chars = sorted(list(set(text)))
    vocab = {ch: i for i, ch in enumerate(chars)}
    CONFIG['vocab_size'] = len(vocab)
    return vocab, {i: ch for i, ch in enumerate(chars)}

def prepare_data():
    # Download Tiny Shakespeare
    shakespeare_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    shakespeare_path = os.path.join(CONFIG['data_dir'], 'shakespeare.txt')
    download_data(shakespeare_url, shakespeare_path)
    
    with open(shakespeare_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    cleaned_text = clean_text(text)
    vocab, inv_vocab = build_vocab(cleaned_text)

    # Save vocab
    torch.save(vocab, os.path.join(CONFIG['data_dir'], 'vocab.pt'))
    torch.save(inv_vocab, os.path.join(CONFIG['data_dir'], 'inv_vocab.pt'))
    
    # Tokenize and save data
    tokens = [vocab[ch] for ch in cleaned_text if ch in vocab]
    torch.save(tokens, os.path.join(CONFIG['data_dir'], 'tokens.pt'))
    
    logger.info("Data prepared.")


# For fine-tuning: Download tiny IMDB subset
def prepare_finetune_data():
    # Mock small IMDB-like data (positive/negative reviews)
    # In reality, download from HF or Kaggle; here, hardcoded for demo
    reviews = [
        ("this movie was great i loved it", 1),
        ("terrible film waste of time", 0),
        ("amazing acting and plot", 1),
        ("boring and predictable", 0),
        # Add more if needed, but keep small
    ] * 100  # Duplicate for volume
    
    tokens_list = []
    labels = []
    vocab = torch.load(os.path.join(CONFIG['data_dir'], 'vocab.pt'))
    
    for review, label in reviews:
        cleaned = clean_text(review)
        tokens = [vocab[ch] for ch in cleaned if ch in vocab]
        tokens_list.append(tokens)
        labels.append(label)
    
    torch.save(tokens_list, os.path.join(CONFIG['data_dir'], 'finetune_tokens.pt'))
    torch.save(labels, os.path.join(CONFIG['data_dir'], 'finetune_labels.pt'))
    
    logger.info("Fine-tune data prepared.")

if __name__ == "__main__":  
    prepare_data()
    prepare_finetune_data()