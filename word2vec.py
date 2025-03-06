import pandas as pd # ramkumar
from gensim.models import Word2Vec

# Load CSV file (Ensure file path is correct)
file_path = r"D:/2ND SEM/delete/converted_sequences_class 1.csv" # Change if needed
df = pd.read_csv(file_path)

# Check if the required columns exist
if "Header" not in df.columns or "Sequence" not in df.columns:
    raise ValueError("CSV file must have 'Header' and 'Sequence' columns.")

# Function to generate k-mers from sequences
def generate_kmers(sequence, k=6):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Apply k-mers transformation
df["Kmers"] = df["Sequence"].apply(lambda x: generate_kmers(str(x), k=6))

# Train Word2Vec model on k-mers
w2v_model = Word2Vec(sentences=df["Kmers"], vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Function to get sequence embedding
import numpy as np
def get_sequence_embedding(sequence, model, k=6):
    kmers = generate_kmers(str(sequence), k)
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Convert sequences into numerical embeddings
df["Embedding"] = df["Sequence"].apply(lambda x: get_sequence_embedding(x, w2v_model))

# Save processed data
output_path = r"D:\2ND SEM\delete\processed_sequences.csv"
df.to_csv(output_path, index=False)

print(f"Processed dataset saved to: {output_path}")

print("hello")
