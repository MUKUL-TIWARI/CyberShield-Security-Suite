import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("📌 Loading dataset...")

df = pd.read_csv("dataset/email_dataset.csv")

# Normalize labels
df["label"] = df["label"].str.lower().str.strip()

# Combine fields into one text column
df["text"] = (
    df["subject"].astype(str) + " " +
    df["body"].astype(str) + " " +
    df["sender"].astype(str)
)

# Balance dataset
phishing = df[df.label == "phishing"]
safe = df[df.label == "safe"]

min_len = min(len(phishing), len(safe))

phishing = phishing.sample(min_len)
safe = safe.sample(min_len)

df = pd.concat([phishing, safe]).sample(frac=1)

print("\n📊 Balanced label counts:")
print(df.label.value_counts())

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=9000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=700,
    max_depth=None,
    n_jobs=-1
)

print("\n🤖 Training model...")
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"\n✅ Final Accuracy: {acc*100:.2f}%")

# Save files
joblib.dump(vectorizer, "email_vectorizer.pkl")
joblib.dump(model, "email_rf_model.pkl")

print("\n🎉 Training Completed!")
