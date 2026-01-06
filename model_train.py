import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load data
df = pd.read_csv("products.csv")

# ---- Normalize Gender ----
def normalize_gender(text):
    text = str(text).lower()

    if "men" in text or "male" in text or text.startswith("m"):
        return "men"
    elif "women" in text or "female" in text or "lady" in text or text.startswith("w"):
        return "women"
    return "unisex"

# df["gender"] = df["gender"].apply(normalize_gender) 

df["gender"] = df["gender"].replace({
    "male": "men", "man": "men", "mens": "men",
    "female": "women", "woman": "women", "ladies": "women"
})


# ---- TEXT PREPARATION ----
df["combined"] = (
    df["name"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["tags"].astype(str) + " " +
    df["categories"].astype(str) + " " +
    df["short_description"].astype(str) + " " +
    df["color"].astype(str)
)

# ---- TARGET COLUMN ----
# You can pick any of these depending on your use case:
target_column = "gender"  # Options: category, color, occasion etc.

X = df["combined"]
y = df[target_column].astype(str)

# ---- Split Training/Test ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- TF-IDF Vectorization ----
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---- TRAIN CLASSIFIER ----
model = RandomForestClassifier(
    n_estimators=200,      # number of trees
    max_depth=None,       # let trees fully grow (can tune later)
    random_state=42
)

model.fit(X_train_vec, y_train)
# ---- EVALUATE ----
predictions = model.predict(X_test_vec)
print(classification_report(y_test, predictions))

# ---- SAVE MODEL + VECTORIZER ----
with open("classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("🔥 Gender Classification Model Trained & Saved!")
