import os
import logging
import torch
import torch.nn as nn
from src.model import MiniGPT
from scripts.fine_tune import MiniGPTForClassification
from scripts.data_prep import clean_text
from src.tokenizer import CharTokenizer
import yaml

with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Set up logging for monitoring (non-functional: observability)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inference():
    model = MiniGPT(CONFIG['vocab_size'], CONFIG['embed_dim'], CONFIG['num_heads'], CONFIG['num_layers'], CONFIG['max_seq_len'])
    model.load_state_dict(torch.load(os.path.join(CONFIG['model_dir'], 'minigpt.pth')))
    model.to(CONFIG['device'])
    
    tokenizer = CharTokenizer(os.path.join(CONFIG['data_dir'], 'vocab.pt'))
    
    prompt = "to be or not to be"
    input_tokens = torch.tensor([tokenizer.encode(prompt)]).to(CONFIG['device'])
    generated = model.generate(input_tokens, max_new_tokens=50)
    output = tokenizer.decode(generated[0].tolist())
    
    logger.info(f"Generated: {output}")

    # Fine-tuned inference (classification)
    ft_model = MiniGPTForClassification(model)
    ft_model.load_state_dict(torch.load(os.path.join(CONFIG['model_dir'], 'minigpt_finetuned.pth')))
    ft_model.to(CONFIG['device'])
    
    test_prompt = "this was amazing"
    input_tokens = torch.tensor([tokenizer.encode(clean_text(test_prompt)) + [0] * (CONFIG['max_seq_len'] - len(test_prompt))]).to(CONFIG['device'])
    mask = nn.Transformer.generate_square_subsequent_mask(input_tokens.size(1)).to(CONFIG['device'])
    with torch.no_grad():
        logits = ft_model(input_tokens, mask)
        pred = torch.argmax(logits, dim=1).item()
    sentiment = "Positive" if pred == 1 else "Negative"
    logger.info(f"Sentiment for '{test_prompt}': {sentiment}")

if __name__ == "__main__":  
    inference()