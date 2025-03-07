#combined traditional physicochemical properties with deep learning embeddings from ProBERT( dataset)

import pandas as pd
import numpy as np
import joblib
from transformers import BertTokenizer, BertModel
import torch
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 📌 Load Dataset
dataset_path = r"D:/2ND SEM/AMR_2nd/protein2_dataset.csv"
df = pd.read_csv(dataset_path)

# 🔍 Check Column Names (Debugging Step)
print("Columns in dataset:", df.columns)

# 📌 Ensure 'Sequence' column exists
if "Sequence" not in df.columns:
    raise KeyError("Column 'Sequence' not found. Check dataset headers!")

# 🔹 Extract Physicochemical Features
def extract_physicochemical_properties(sequence):
    """Extract simple physicochemical properties"""
    length = len(sequence)
    hydrophobicity = sum(1 for aa in sequence if aa in "AILMFWYV") / length  # Fraction of hydrophobic AAs
    charge = sum(1 for aa in sequence if aa in "KRH") - sum(1 for aa in sequence if aa in "DE")  # Net charge
    return [length, hydrophobicity, charge]

df["physicochemical"] = df["Sequence"].apply(extract_physicochemical_properties)
physico_features = pd.DataFrame(df["physicochemical"].tolist(), columns=["Length", "Hydrophobicity", "Charge"])

# 📌 Load ProBERT Model & Tokenizer
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
model = BertModel.from_pretrained("Rostlab/prot_bert")

# 🔹 Function to Extract ProBERT Embeddings
def get_probert_embedding(sequence):
    tokens = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

df["probert_embedding"] = df["Sequence"].apply(get_probert_embedding)
probert_features = pd.DataFrame(df["probert_embedding"].tolist())

# 🔹 Combine All Features
final_features = pd.concat([physico_features, probert_features], axis=1)
X = final_features
y = df["Label"]  # Assuming 'Label' is the target column

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train XGBoost Hybrid Model
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# 📌 Save Trained Model
joblib.dump(model, "D:/2ND SEM/delete/NEW/mar6 - proteins/xgboost_model.pkl")

# 🔹 Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# 📌 Save Processed Dataset with Features
final_features["Label"] = y
final_features.to_csv("D:/2ND SEM/delete/NEW/mar6 - proteins/protein2_dataset_with_embeddings.csv", index=False)

print("✅ Hybrid Model Training Completed & Saved!")
