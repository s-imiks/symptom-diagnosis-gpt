from fastapi import FastAPI
import torch
from src.model import GPTModel
from src.config import config

app = FastAPI(title="Symptom Diagnosis GPT API")
model = GPTModel(vocab_size=200)
model.load_state_dict(torch.load("data/processed/model.pt", map_location=config['device']))
model.eval()

@app.post("/predict")
def predict(symptoms: dict):
    text = symptoms.get("text", "")
    return {"diagnosis": f"Model suggests: [mock diagnosis for '{text}']"}
