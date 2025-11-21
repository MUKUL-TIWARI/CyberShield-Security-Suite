from flask import Flask, render_template, request, jsonify
import os
import re
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)

# -------------------------------
# LOAD EMAIL + CALL MODELS
# -------------------------------
email_model = joblib.load("email_rf_model.pkl")
email_vectorizer = joblib.load("email_vectorizer.pkl")

call_model = joblib.load("call_rf_model.pkl")
call_vectorizer = joblib.load("call_vectorizer.pkl")

# -------------------------------
# LOAD URL MODEL + VECTOR + SCALER
# -------------------------------
try:
    url_model = joblib.load("phishing.pkl")
    url_vectorizer = joblib.load("vectorizer.pkl")
    url_scaler = joblib.load("url_scaler.pkl") 
    url_numeric_features = joblib.load("url_numeric_features.pkl")
    print("✅ URL ML Model Loaded Successfully")
except Exception as e:
    print("❌ URL model failed:", e)
    url_model = None
    url_vectorizer = None
    url_scaler = None



# --- feature extraction must match training ---
def _count_dots(url): return url.count('.')
def _url_length(url): return len(url)
def _has_ip(url): return 1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", url) else 0
def _contains_https(url): return 1 if "https://" in url.lower() else 0
def _suspicious_keywords(url):
    keywords = ["login","verify","secure","account","update","bank"]
    return sum(1 for k in keywords if k in url.lower())

def _extract_numeric_features(url):
    """
    Create numeric features in the same order used during training:
    ['dots', 'length', 'ip', 'https', 'keywords']
    """
    dots = _count_dots(url)
    length = _url_length(url)
    ip = _has_ip(url)
    https = _contains_https(url)
    keywords = _suspicious_keywords(url)
    arr = np.array([[dots, length, ip, https, keywords]], dtype=float)
    return arr

def predict_url(url):
    url = (url or "").strip()
    if not url:
        return "No URL provided."

    # -------------------------------
    # 1. WHITELIST (for demo stability)
    # -------------------------------
    SAFE_DOMAINS = {
        "google.com", "www.google.com",
        "github.com", "www.github.com",
        "microsoft.com", "www.microsoft.com",
        "python.org", "openai.com"
    }

    try:
        # Extract hostname
        host = url.split("://")[-1].split("/")[0].split(":")[0].lower()
        if host in SAFE_DOMAINS:
            return "Safe: Trusted domain (whitelisted)."
    except:
        pass

    # -------------------------------
    # 2. If ML model is missing → fallback
    # -------------------------------
    if url_model is None or url_vectorizer is None or url_scaler is None:
        return heuristic_url_result(url)  # fallback to heuristic

    # -------------------------------
    # 3. Extract features (numeric + text)
    # -------------------------------
    # TF-IDF
    try:
        X_text = url_vectorizer.transform([url])
    except Exception as e:
        print("Vectorizer error:", e)
        return heuristic_url_result(url)

    # Numeric features
    try:
        X_num = _extract_numeric_features(url)
        X_num_scaled = url_scaler.transform(X_num)
        X_num_sparse = csr_matrix(X_num_scaled)
    except Exception as e:
        print("Scaler error:", e)
        return heuristic_url_result(url)

    # Combine features
    from scipy.sparse import hstack
    X = hstack([X_text, X_num_sparse])

    # -------------------------------
    # 4. Model prediction + probabilities
    # -------------------------------
    try:
        pred = url_model.predict(X)[0]
        proba = url_model.predict_proba(X)[0]

        # Get BAD class probability
        idx_bad = list(url_model.classes_).index("bad")
        bad_prob = proba[idx_bad]  # between 0 and 1

    except Exception as e:
        print("Model prediction error:", e)
        return heuristic_url_result(url)

    # -------------------------------
    # 5. Decision based on confidence
    # -------------------------------
    if bad_prob >= 0.80:
        return f"Danger: Likely phishing URL. (Confidence: {bad_prob*100:.1f}%)"

    elif bad_prob >= 0.55:
        return f"Warning: Suspicious signs detected. (Confidence: {bad_prob*100:.1f}%)"

    else:
        return f"Safe: URL appears legitimate. (Risk: {bad_prob*100:.1f}%)"



# -------------------------------------------------------
# HEURISTIC FALLBACK FUNCTION (kept separate & clean)
# -------------------------------------------------------
def heuristic_url_result(url):
    suspicious_keywords = [
        'login', 'update', 'secure', 'bank', 'verify', 'confirm',
        'account', 'password', 'reset', 'otp'
    ]

    score = 0

    if re.search(r"(\d{1,3}\.){3}\d{1,3}", url):
        score += 2
    if len(url) > 75:
        score += 1
    for k in suspicious_keywords:
        if k in url.lower():
            score += 2
    if url.count('.') > 4:
        score += 1
    if 'https://' not in url.lower():
        score += 1

    if score >= 3:
        return f"Danger (Heuristic): Signs of phishing. Score={score}"
    elif score == 2:
        return f"Warning (Heuristic): Possibly suspicious. Score={score}"
    else:
        return f"Safe (Heuristic): Low risk. Score={score}"
    
# -------------------------------
# EMAIL PREDICTOR
# -------------------------------
def predict_email_ml(subject, body, sender):
    text = f"{subject} {body} {sender}"
    X = email_vectorizer.transform([text])
    pred = email_model.predict(X)[0]
    return "Phishing Email" if pred == "phishing" else "Safe Email"


# -------------------------------
# CALL PREDICTOR
# -------------------------------
def predict_call_ml(transcript):
    X = call_vectorizer.transform([transcript])
    pred = call_model.predict(X)[0]
    return "Scam Call" if pred == "scam" else "Safe Call"


# -------------------------------
# FLASK ROUTES
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    url_input = ""
    email_subject = ""
    email_body = ""
    email_sender = ""
    call_transcript = ""
    call_number = ""

    url_result = ""
    email_result = ""
    call_result = ""

    if request.method == 'POST':
        url_input = request.form.get('url_input', '').strip()
        email_subject = request.form.get('email_subject', '').strip()
        email_body = request.form.get('email_body', '').strip()
        email_sender = request.form.get('email_sender', '').strip()
        call_transcript = request.form.get('call_transcript', '').strip()
        call_number = request.form.get('call_number', '').strip()

        if url_input:
            url_result = predict_url(url_input)

        if (email_subject or email_body or email_sender):
            email_result = predict_email_ml(email_subject, email_body, email_sender)

        if call_transcript:
            call_result = predict_call_ml(call_transcript)

    return render_template(
        'index.html',
        url_input=url_input,
        predict=url_result,
        email_subject=email_subject,
        email_body=email_body,
        email_sender=email_sender,
        call_transcript=call_transcript,
        call_number=call_number,
        url_result=url_result,
        email_result=email_result,
        call_result=call_result
    )
@app.route("/scan_url", methods=["POST"])
def scan_url():
    url = request.json.get("url", "")
    result = predict_url(url)
    return jsonify({"result": result})

@app.route("/scan_email", methods=["POST"])
def scan_email():
    subject = request.json.get("subject", "")
    sender = request.json.get("sender", "")
    body = request.json.get("body", "")
    result = predict_email_ml(subject, body, sender)
    return jsonify({"result": result})

@app.route("/scan_call", methods=["POST"])
def scan_call():
    transcript = request.json.get("transcript", "")
    result = predict_call_ml(transcript)
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(debug=True)
