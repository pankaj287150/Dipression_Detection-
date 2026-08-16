"""
app.py
======
Flask backend for the Depression Detection system.

Endpoints:
  GET  /              → serves index.html
  POST /predict       → JSON {text: "..."} → prediction + confidence
  GET  /models/status → loaded model names + readiness

Run:
  python app.py
  Listening on http://localhost:5000
"""

import os
import pickle
import re
import json
import logging

import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer
from scipy.special import softmax
from nltk.corpus import stopwords

# ─── local imports ───────────────────────────────────────────
from models import BertClassifier, DEVICE

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# ─────────────────────────────────────────────
# Paths  (adjust to your Drive mount or local)
# ─────────────────────────────────────────────
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/content/drive/MyDrive/depression_detection/saved_models"
)

# ─────────────────────────────────────────────
# Pre-load all models at startup
# ─────────────────────────────────────────────
import nltk
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 1)


class ModelStore:
    def __init__(self):
        self.vectorizer  = None
        self.ml_models   = {}       # {name: sklearn model}
        self.dl_models   = {}       # {name: torch model}
        self.bert_tok    = None
        self._loaded     = []

    def load_all(self):
        logging.info("Loading models from %s", MODEL_DIR)

        # TF-IDF vectorizer
        vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
        if os.path.exists(vec_path):
            self.vectorizer = pickle.load(open(vec_path, "rb"))
            self._loaded.append("TF-IDF Vectorizer")

        # ML models
        for name in ["LR", "SVM", "RandomForest", "ensemble"]:
            path = os.path.join(MODEL_DIR, f"{name}.pkl")
            if os.path.exists(path):
                self.ml_models[name] = pickle.load(open(path, "rb"))
                self._loaded.append(name)

        # BERT tokenizer
        try:
            self.bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            logging.warning("BERT tokenizer not available — DL inference disabled.")

        # DL models — BERT only
        dl_classes = {
            "BERT": BertClassifier,
        }
        for name, cls in dl_classes.items():
            path = os.path.join(MODEL_DIR, f"{name}.pt")
            if os.path.exists(path):
                m = cls().to(DEVICE)
                m.load_state_dict(torch.load(path, map_location=DEVICE))
                m.eval()
                self.dl_models[name] = m
                self._loaded.append(name)

        logging.info("Loaded: %s", self._loaded)

    @property
    def ready(self):
        return bool(self.ml_models or self.dl_models)


store = ModelStore()
store.load_all()


# ─────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────
def predict_ml(text: str, model_name: str = "ensemble"):
    """TF-IDF + sklearn model."""
    cleaned = clean_text(text)
    vec     = store.vectorizer.transform([cleaned])
    model   = store.ml_models[model_name]
    pred    = int(model.predict(vec)[0])
    conf    = float(model.predict_proba(vec)[0][pred])
    return pred, conf


def predict_dl(text: str, model_name: str = "BERT"):
    """BERT tokenizer + PyTorch model."""
    enc = store.bert_tok(
        text, padding='max_length', truncation=True,
        max_length=128, return_tensors='pt'
    )
    with torch.no_grad():
        ids  = enc['input_ids'].to(DEVICE)
        mask = enc['attention_mask'].to(DEVICE)
        logits = store.dl_models[model_name](ids, mask).cpu().numpy()
    probs = softmax(logits, axis=1)[0]
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return pred, conf


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: {"text": "...", "model": "ensemble"}
    Returns:      {"prediction": "Depressed", "confidence": 0.92,
                   "label": 1, "model_used": "ensemble"}
    """
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    model_name = body.get("model", "ensemble")

    try:
        # Prefer the requested model
        if model_name in store.dl_models and store.bert_tok:
            pred, conf = predict_dl(text, model_name)
            used = model_name + " (DL)"
        elif model_name in store.ml_models and store.vectorizer:
            pred, conf = predict_ml(text, model_name)
            used = model_name + " (ML)"
        elif store.ml_models and store.vectorizer:
            fallback = "ensemble" if "ensemble" in store.ml_models else list(store.ml_models.keys())[0]
            pred, conf = predict_ml(text, fallback)
            used = fallback + " (ML-fallback)"
        else:
            return jsonify({"error": "No models available"}), 503

        label = "Depressed" if pred == 1 else "Not Depressed"
        return jsonify({
            "prediction" : label,
            "label"      : pred,
            "confidence" : round(conf, 4),
            "model_used" : used,
        })

    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/models/status")
def models_status():
    return jsonify({
        "ready"       : store.ready,
        "loaded"      : store._loaded,
        "ml_models"   : list(store.ml_models.keys()),
        "dl_models"   : list(store.dl_models.keys()),
        "vectorizer"  : store.vectorizer is not None,
        "bert_tokenizer": store.bert_tok is not None,
    })


@app.route("/models/list")
def models_list():
    """Return all available model names for the frontend dropdown."""
    all_models = list(store.ml_models.keys()) + list(store.dl_models.keys())
    return jsonify({"models": all_models})


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
