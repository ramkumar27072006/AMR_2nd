#Example model for protein sequence virulent Prediction
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 📌 Load Dataset
file_path = "D:/2ND SEM/delete/NEW/mar6 - proteins/protein2_dataset_with_embeddings.csv"
df = pd.read_csv(file_path)

# 📌 Extract Features & Labels
X = df.drop(columns=["Label"])  # Drop target column
y = df["Label"]

# 📌 Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train XGBoost Model
model = xgb.XGBClassifier(
    n_estimators=200, 
    max_depth=6,       
    learning_rate=0.1, 
    eval_metric="mlogloss",
    use_label_encoder=False
)
model.fit(X_train, y_train)

# 📌 Save Model as `.txt` File
model_txt_path = "D:/2ND SEM/delete/NEW/mar6 - proteins/xgboost_model.txt"
model.save_model(model_txt_path)

# 🔹 Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print(f"📄 Model saved as: {model_txt_path}")

# -------------------------------
# 🔥 User Input for New Prediction
# -------------------------------
def extract_physicochemical_properties(sequence):
    """Extract simple physicochemical properties"""
    length = len(sequence)
    hydrophobicity = sum(1 for aa in sequence if aa in "AILMFWYV") / length  # Fraction of hydrophobic AAs
    charge = sum(1 for aa in sequence if aa in "KRH") - sum(1 for aa in sequence if aa in "DE")  # Net charge
    return [length, hydrophobicity, charge]

def predict_virulence(sequence):
    """Predict virulence for a new protein sequence"""
    # Extract features
    features = extract_physicochemical_properties(sequence)
    
    # Check if the dataset has embeddings (if yes, use mean embeddings)
    if "0" in df.columns:  # Assuming embeddings start from column index 0
        num_embedding_features = sum(c.isdigit() for c in df.columns)
        embedding_values = [0] * num_embedding_features  # Placeholder for embeddings (can integrate ProBERT later)
        features.extend(embedding_values)
    
    # Convert to NumPy array and reshape
    features = np.array(features).reshape(1, -1)

    # Load trained model
    model = xgb.XGBClassifier()
    model.load_model(model_txt_path)

    # Predict
    prediction = model.predict(features)[0]
    result = "Virulent" if prediction == 1 else "Non-Virulent"
    
    return result

# 📌 Take User Input
user_sequence = input("\n🔍 Enter a new protein sequence: ")
prediction_result = predict_virulence(user_sequence)
print(f"\n🧬 Prediction Result: {prediction_result}")
