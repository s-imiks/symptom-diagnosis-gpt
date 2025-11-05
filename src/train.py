import torch
import torch.nn as nn
import torch.optim as optim
from src.model import GPTModel
from src.config import config

def train_model(vocab_size=200):
    model = GPTModel(vocab_size).to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    # Dummy example for testing
    x = torch.randint(0, vocab_size, (config['batch_size'], config['block_size'])).to(config['device'])
    y = torch.randint(0, vocab_size, (config['batch_size'], config['block_size'])).to(config['device'])

    for step in range(100):
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "data/processed/model.pt")
    print("✅ Model saved to data/processed/model.pt")

if __name__ == "__main__":
    train_model()
