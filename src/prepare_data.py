import os

def build_dataset():
    # Example: small placeholder dataset
    data = [
        "I have fever and cough. Diagnosis: flu.",
        "I have chest pain and shortness of breath. Diagnosis: heart disease.",
        "I have headache and sore throat. Diagnosis: cold."
    ]
    os.makedirs('data/processed', exist_ok=True)
    with open('data/dataset.txt', 'w') as f:
        f.write("\n".join(data))
    print("✅ dataset.txt created in data/")

if __name__ == "__main__":
    build_dataset()
