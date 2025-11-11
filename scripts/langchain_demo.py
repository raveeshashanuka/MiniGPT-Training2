import os
import logging
import torch
from src.model import MiniGPT
from src.tokenizer import CharTokenizer
import yaml
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Set up logging for monitoring (non-functional: observability)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Integration with LangChain for prompt engineering/experimentation
class CustomLLM(LLM):
    model: MiniGPT
    tokenizer: CharTokenizer
    device: str
    
    def _call(self, prompt: str, stop=None) -> str:
        input_tokens = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        generated = self.model.generate(input_tokens, max_new_tokens=50, temperature=0.8)
        return self.tokenizer.decode(generated[0].tolist())[len(prompt):]
    
    @property
    def _llm_type(self) -> str:
        return "custom_minigpt"

def langchain_demo():
    model = MiniGPT(CONFIG['vocab_size'], CONFIG['embed_dim'], CONFIG['num_heads'], CONFIG['num_layers'], CONFIG['max_seq_len'])
    model.load_state_dict(torch.load(os.path.join(CONFIG['model_dir'], 'minigpt.pth')))
    model.to(CONFIG['device'])
    
    tokenizer = CharTokenizer(os.path.join(CONFIG['data_dir'], 'vocab.pt'))
    
    llm = CustomLLM(model=model, tokenizer=tokenizer, device=CONFIG['device'])
    
    template = PromptTemplate(input_variables=["topic"], template="Write a short poem about {topic} in Shakespeare style.")
    chain = LLMChain(llm=llm, prompt=template)
    
    output = chain.run("love")
    logger.info(f"LangChain Output: {output}")

if __name__ == "__main__":  # For langchain_demo.py
    langchain_demo()