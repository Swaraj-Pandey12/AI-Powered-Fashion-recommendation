import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load metadata
df = pd.read_csv("data/filtered_metadata.csv")

# Ensure gender labels are string
df["gender"] = df["gender"].astype(str)

# Combine text for training (name + description)
df["text"] = (
    df["name"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["tags"].fillna("")
)

# Only keep rows with gender set
df = df[df["gender"].str.strip() != ""]

X = df["text"]
y = df["gender"]

# Build pipeline: TFIDF + RandomForest
clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train
clf.fit(X, y)

# Save the model
with open("models/gender_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("RandomForest gender classifier trained and saved as models/gender_model.pkl")
