import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/metadata.csv")

# Fill NaN values
df.fillna("", inplace=True)

# Combine important descriptive fields for better embeddings
df["combined_text"] = (
    df["name"].astype(str) + " " +
    df["short_description"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["tags"].astype(str) + " " +
    df["categories"].astype(str) + " " +
    df["occasions"].astype(str) + " " +
    df["designer"].astype(str) + " " +
    df["material"].astype(str) + " " +
    df["color"].astype(str)
)

print("Number of products:", len(df))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
texts = df["combined_text"].tolist()
embeddings = model.encode(texts, show_progress_bar=True)

# Convert to float32
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "data/product_index.faiss")

# Also save filtered metadata to keep alignment
df.to_csv("data/filtered_metadata.csv", index=False)

print("FAISS index built successfully")
