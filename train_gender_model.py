# ==============================
# 1. IMPORT REQUIRED LIBRARIES
# ==============================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ==============================
# 2. LOAD DATA
# ==============================

df = pd.read_csv("data/filtered_metadata.csv")

# 🔥 DROP NaN gender FIRST (IMPORTANT)
df = df[df["gender"].notna()]

# Ensure gender is string
df["gender"] = df["gender"].astype(str)

# Combine text features
df["text"] = (
    df["name"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["tags"].fillna("")
)

# Remove empty gender rows
df = df[df["gender"].str.strip() != ""]

print("\n✅ Dataset loaded successfully")


# ==============================
# 3. CHECK CLASS DISTRIBUTION
# ==============================

print("\n📊 Class Distribution (Count):")
print(df["gender"].value_counts())

print("\n📊 Class Distribution (Percentage):")
print(df["gender"].value_counts(normalize=True) * 100)


# ==============================
# 4. SPLIT DATA
# ==============================

X = df["text"]
y = df["gender"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # VERY IMPORTANT for imbalanced data
)

print("\n✅ Train-test split completed")


# ==============================
# 5. BUILD PIPELINE
# ==============================

pipeline = ImbPipeline([
    # Converts text → numerical vectors
    ("tfidf", TfidfVectorizer(
        max_features=5000,   # limits vocabulary size to reduce noise
        ngram_range=(1, 2)   # unigrams + bigrams (better context)
    )),

    # Oversampling minority class (Men)
    ("smote", SMOTE(random_state=42)),

    # Random Forest classifier
    ("rf", RandomForestClassifier(
        n_estimators=200,        # number of trees
        random_state=42,
        class_weight="balanced", # extra penalty for minority class
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
    ))
])




# ==============================
# 6. TRAIN MODEL
# ==============================

pipeline.fit(X_train, y_train)

print("\n✅ Model training completed")


# ==============================
# 7. EVALUATE MODEL
# ==============================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Model Accuracy:", accuracy)

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))


# ==============================
# 8. SAVE MODEL
# ==============================

with open("models/gender_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\n💾 Model saved as models/gender_model.pkl")
