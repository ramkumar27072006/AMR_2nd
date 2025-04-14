import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from scipy.stats import entropy
from collections import Counter
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv(r"C:\Users\R RAMKUMAR\Downloads\Gene sequence Dataset.csv", usecols=["DNA Sequence", "Class"])
df.dropna(inplace=True)
df.rename(columns={"DNA Sequence": "sequence", "Class": "label"}, inplace=True)

# Biological features
def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)

def shannon_entropy(seq):
    prob = [n_x / len(seq) for x, n_x in Counter(seq).items()]
    return entropy(prob, base=2)

df["gc_content"] = df["sequence"].apply(gc_content)
df["length"] = df["sequence"].apply(len)
df["entropy"] = df["sequence"].apply(shannon_entropy)

# Generate 6-mers
def generate_kmers(seq, k=6):
    return ' '.join([seq[i:i+k] for i in range(len(seq) - k + 1)])

df["kmers"] = df["sequence"].apply(lambda x: generate_kmers(x, 6))

# TF-IDF Vectorization
tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=8000)
X_tfidf = tfidf.fit_transform(df["kmers"])

# Combine with handcrafted features
X_handcrafted = df[["gc_content", "length", "entropy"]]
X_handcrafted_scaled = StandardScaler().fit_transform(X_handcrafted)

# Combine all features
X = hstack([X_tfidf, X_handcrafted_scaled])
y = LabelEncoder().fit_transform(df["label"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
