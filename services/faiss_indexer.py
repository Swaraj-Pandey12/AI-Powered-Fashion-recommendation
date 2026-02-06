import faiss
import os
import pandas as pd

class FaissIndexer:

    def __init__(self, index_path: str, metadata_path: str):

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load filtered metadata
        self.metadata = pd.read_csv(metadata_path)

    def search(self, vector, top_k: int = 20):
        distances, indices = self.index.search(vector, top_k)
        results = []

        for idx in indices[0]:
            if idx < len(self.metadata):
                product = self.metadata.iloc[idx].to_dict()
                results.append(product)

        return results
