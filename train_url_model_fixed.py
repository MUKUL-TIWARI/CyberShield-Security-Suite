import pandas as pd
import re
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

print("📌 Loading dataset...")
df = pd.read_csv("dataset/phishing_site_urls.csv")

# Standardize column names
df.rename(columns={"url": "URL", "type": "Label"}, inplace=True)
df["Label"] = df["Label"].str.lower().str.strip()

# Map all bad categories to "bad"
df["Label"] = df["Label"].replace({
    "benign": "good",
    "legitimate": "good",
    "safe": "good",
    "phishing": "bad",
    "malware": "bad",
    "defacement": "bad"
})

# Remove weird/missing URLs
df = df[df["URL"].notna()]

# Balance using class weights (NO sampling)
print(df["Label"].value_counts())

# --------------------------
# IMPROVED NUMERIC FEATURES
# --------------------------
def count_dots(url): return url.count('.')
def url_length(url): return len(url)
def count_hyphens(url): return url.count('-')
def count_slashes(url): return url.count('/')
def has_ip(url): return 1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", url) else 0

def count_suspicious_words(url):
    keywords = [
        "secure", "account", "update", "verify", "login", 
        "bank", "signin", "confirm", "reset", "password"
    ]
    return sum(1 for k in keywords if k in url.lower())

# Apply numeric features
df["dots"] = df["URL"].apply(count_dots)
df["length"] = df["URL"].apply(url_length)
df["hyphens"] = df["URL"].apply(count_hyphens)
df["slashes"] = df["URL"].apply(count_slashes)
df["ip"] = df["URL"].apply(has_ip)
df["keywords"] = df["URL"].apply(count_suspicious_words)

numeric_features = ["dots", "length", "hyphens", "slashes", "ip", "keywords"]

# --------------------------
# TF-IDF VECTORIZATION
# --------------------------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    token_pattern=r"[A-Za-z0-9:/._?-]+"
)

X_text = vectorizer.fit_transform(df["URL"])

# Numeric array
X_num = df[numeric_features].values

# Scale numeric features
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Convert numeric to sparse & combine
X_num_sparse = csr_matrix(X_num_scaled)
X = hstack([X_text, X_num_sparse])

y = df["Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# --------------------------
# RANDOM FOREST MODEL
# --------------------------
print("🤖 Training improved model...")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    n_jobs=-1,
    class_weight={"good": 1.0, "bad": 1.0}  # balanced learning
)

model.fit(X_train, y_train)

# --------------------------
# EVALUATION
# --------------------------
print("\n📊 Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_train[:len(y_test)], zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))

# --------------------------
# SAVE ARTIFACTS
# --------------------------
joblib.dump(model, "phishing.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(scaler, "url_scaler.pkl")
joblib.dump(numeric_features, "url_numeric_features.pkl")

print("\n🎉 Training Complete — New Model Saved Successfully!")