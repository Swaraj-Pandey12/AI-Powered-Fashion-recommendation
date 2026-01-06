import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

df = pd.read_csv("products.csv")

df["combined"] = (
    df["name"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["tags"].astype(str) + " " +
    df["Catogories"].astype(str)+ " " +
    df["color"].astype(str)
)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["combined"].tolist(), convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "product_index.faiss")
pickle.dump(df, open("products.pkl", "wb"))
print("🚀 FAISS index ready!")
    