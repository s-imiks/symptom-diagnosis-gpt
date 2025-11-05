import torch
import torch.nn as nn
from src.config import config

class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, config['n_embd'])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['n_embd'],
                nhead=config['n_head'],
                dim_feedforward=config['n_embd']*4,
                dropout=config['dropout']
            ),
            num_layers=config['n_layer']
        )
        self.ln = nn.LayerNorm(config['n_embd'])
        self.fc = nn.Linear(config['n_embd'], vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.fc(x)
        return logits
