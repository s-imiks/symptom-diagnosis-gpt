# 🧠 Symptom-Diagnosis-GPT
A simple transformer-based (nanoGPT-style) model that takes in text symptoms and predicts possible diagnoses.

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
uvicorn src.api:app --reload
```
