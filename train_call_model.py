import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load call dataset from dataset folder
df = pd.read_csv("dataset/call_dataset.csv")

# Columns must be: transcript, label
X = df["transcript"]
y = df["label"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Test model
pred = model.predict(X_test)
print("CALL MODEL ACCURACY:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save
joblib.dump(model, "call_rf_model.pkl")
joblib.dump(vectorizer, "call_vectorizer.pkl")

print("Call model saved successfully!")
