"""
train.py
========
Full end-to-end training pipeline:
  1. Load & split data
  2. TF-IDF + ML baseline (before SMOTE)
  3. SMOTE balancing → ML re-train
  4. BERT / BERT+LSTM / BERT+BiLSTM training
  5. Ensemble evaluation
  6. All graphs: ROC-AUC, Decision Curve, Accuracy Comparison, DL Loss/Acc
  7. Save models & vectorizer to Google Drive
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              roc_curve, auc, confusion_matrix,
                              ConfusionMatrixDisplay, classification_report)
from imblearn.over_sampling import SMOTE
from scipy.special import softmax

from preprocessing import load_data, split_data
from models import get_ml_models, get_dl_models, BertClassifier, DEVICE, BERT_NAME

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 100, 'font.size': 11})

# ─────────────────────────────────────────────
# Google Drive paths  (edit these)
# ─────────────────────────────────────────────
DRIVE_ROOT  = "/content/drive/MyDrive/depression_detection"
PATH1       = f"{DRIVE_ROOT}/balanced_dataset.csv"
PATH2       = f"{DRIVE_ROOT}/imbalanced_dataset.csv"
MODEL_DIR   = f"{DRIVE_ROOT}/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
BERT_MAX_LEN   = 128
BERT_BATCH     = 16
BERT_EPOCHS    = 3
BERT_LR        = 2e-5
TFIDF_FEATURES = 10_000
RANDOM_STATE   = 42


# ═══════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════
class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long)
        }


def plot_confusion(y_true, y_pred, title, save_path):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Dep.", "Depressed"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ═══════════════════════════════════════════════════════════════
# ROC curves  (multi-model on one plot)
# ═══════════════════════════════════════════════════════════════
def plot_roc_multi(roc_data: dict, title: str, save_path: str):
    """
    roc_data = {model_name: (fpr, tpr, auc_score)}
    """
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name}  AUC={auc_val:.3f}", lw=2)
    plt.plot([0,1],[0,1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ═══════════════════════════════════════════════════════════════
# Decision Curve Analysis (DCA)
# ═══════════════════════════════════════════════════════════════
def plot_decision_curve(y_true, proba_dict: dict, title: str, save_path: str):
    """
    Plots net benefit vs threshold for multiple models.
    proba_dict = {model_name: prob_array}
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    n = len(y_true)
    y = np.array(y_true)

    plt.figure(figsize=(9, 6))

    # Treat-all baseline
    treat_all = [
        (y.sum()/n) - (1 - y.sum()/n) * (t/(1-t))
        for t in thresholds
    ]
    plt.plot(thresholds, treat_all, 'k-', label='Treat All', lw=1.5)
    plt.axhline(0, color='grey', linestyle='--', lw=1, label='Treat None')

    colors = plt.cm.tab10.colors
    for i, (name, proba) in enumerate(proba_dict.items()):
        net_benefit = []
        for t in thresholds:
            preds     = (proba >= t).astype(int)
            tp        = ((preds == 1) & (y == 1)).sum()
            fp        = ((preds == 1) & (y == 0)).sum()
            nb        = tp/n - fp/n * (t/(1-t))
            net_benefit.append(nb)
        plt.plot(thresholds, net_benefit,
                 label=name, lw=2, color=colors[i % 10])

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ═══════════════════════════════════════════════════════════════
# Accuracy comparison bar chart
# ═══════════════════════════════════════════════════════════════
def plot_accuracy_comparison(results: dict, save_path: str):
    """
    results = {model_name: accuracy}
    """
    names  = list(results.keys())
    accs   = [results[n] * 100 for n in names]
    colors = ['#4CAF50' if a == max(accs) else '#2196F3' for a in accs]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, accs, color=colors, edgecolor='black')
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha='center', fontsize=9)
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ═══════════════════════════════════════════════════════════════
# DL Training loop
# ═══════════════════════════════════════════════════════════════
def train_dl_model(model, train_loader, val_loader, epochs=3, lr=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                    start_factor=1.0, end_factor=0.1, total_iters=epochs)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc' : [], 'val_acc' : []}

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for batch in train_loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            y    = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item() * len(y)
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total   += len(y)

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                y    = batch['label'].to(DEVICE)
                logits = model(ids, mask)
                loss   = criterion(logits, y)
                v_loss    += loss.item() * len(y)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total   += len(y)

        scheduler.step()

        tr_loss = t_loss / t_total
        tr_acc  = t_correct / t_total
        vl_loss = v_loss / v_total
        vl_acc  = v_correct / v_total

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f"  Epoch {epoch+1}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

    return history


def evaluate_dl_model(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].numpy()
            logits = model(ids, mask).cpu().numpy()
            probs  = softmax(logits, axis=1)[:, 1]
            preds  = logits.argmax(axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_dl_history(history, model_name, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   'ro-', label='Val Loss')
    ax1.set_title(f"{model_name} — Loss")
    ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'bo-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'],   'ro-', label='Val Acc')
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ═══════════════════════════════════════════════════════════════
# ██  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Step 1 : Load data ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Loading & Splitting Data")
    print("="*60)
    df                       = load_data(PATH1, PATH2)
    train_df, val_df, test_df = split_data(df)

    X_train_raw = train_df['text'].values
    X_val_raw   = val_df['text'].values
    X_test_raw  = test_df['text'].values
    y_train     = train_df['label'].values
    y_val       = val_df['label'].values
    y_test      = test_df['label'].values

    # ── Step 2 : TF-IDF ────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — TF-IDF Vectorization")
    print("="*60)
    vectorizer   = TfidfVectorizer(max_features=TFIDF_FEATURES, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_val_tfidf   = vectorizer.transform(X_val_raw)
    X_test_tfidf  = vectorizer.transform(X_test_raw)

    # ── Step 3 : ML Baseline (before SMOTE) ────────────────────
    print("\n" + "="*60)
    print("STEP 3 — ML Baseline (Before SMOTE)")
    print("="*60)
    lr, svm, rf, ensemble = get_ml_models()
    ml_models = {"LR": lr, "SVM": svm, "RandomForest": rf}

    roc_before   = {}
    proba_before = {}
    acc_all      = {}

    for name, model in ml_models.items():
        print(f"\nTraining {name}…")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        probs = model.predict_proba(X_test_tfidf)[:, 1]

        acc  = accuracy_score(y_test, preds)
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = auc(fpr, tpr)

        print(f"  Accuracy : {acc:.4f}   AUC : {auc_val:.4f}")
        print(classification_report(y_test, preds,
              target_names=["Not Depressed","Depressed"]))

        roc_before[name]   = (fpr, tpr, auc_val)
        proba_before[name] = probs
        acc_all[f"{name}\n(before)"] = acc

        plot_confusion(y_test, preds,
                       f"{name} — Before SMOTE",
                       f"{MODEL_DIR}/{name}_cm_before.png")

    plot_roc_multi(roc_before,
                   "ROC Curves — ML Models Before SMOTE",
                   f"{MODEL_DIR}/roc_before_smote.png")

    plot_decision_curve(y_test, proba_before,
                        "Decision Curve — Before SMOTE",
                        f"{MODEL_DIR}/dca_before_smote.png")

    # ── Step 4 : SMOTE ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — Applying SMOTE on Training Data")
    print("="*60)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_tfidf, y_train)
    print(f"  Before SMOTE → {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  After  SMOTE → {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

    # ── Step 5 : ML After SMOTE ────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — ML Models After SMOTE")
    print("="*60)
    lr2, svm2, rf2, ensemble = get_ml_models()
    ml_models2 = {"LR": lr2, "SVM": svm2, "RandomForest": rf2}

    roc_after   = {}
    proba_after = {}

    for name, model in ml_models2.items():
        print(f"\nTraining {name} (post-SMOTE)…")
        model.fit(X_train_sm, y_train_sm)
        preds = model.predict(X_test_tfidf)
        probs = model.predict_proba(X_test_tfidf)[:, 1]

        acc  = accuracy_score(y_test, preds)
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = auc(fpr, tpr)

        print(f"  Accuracy : {acc:.4f}   AUC : {auc_val:.4f}")
        print(classification_report(y_test, preds,
              target_names=["Not Depressed","Depressed"]))

        roc_after[name]   = (fpr, tpr, auc_val)
        proba_after[name] = probs
        acc_all[f"{name}\n(after)"] = acc

        plot_confusion(y_test, preds,
                       f"{name} — After SMOTE",
                       f"{MODEL_DIR}/{name}_cm_after.png")

    # ── Step 6 : Ensemble ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6 — Ensemble (Voting Classifier) After SMOTE")
    print("="*60)
    ensemble.fit(X_train_sm, y_train_sm)
    ens_preds = ensemble.predict(X_test_tfidf)
    ens_probs = ensemble.predict_proba(X_test_tfidf)[:, 1]
    ens_acc   = accuracy_score(y_test, ens_preds)
    ens_fpr, ens_tpr, _ = roc_curve(y_test, ens_probs)
    ens_auc   = auc(ens_fpr, ens_tpr)

    print(f"  Ensemble Accuracy : {ens_acc:.4f}   AUC : {ens_auc:.4f}")
    print(classification_report(y_test, ens_preds,
          target_names=["Not Depressed","Depressed"]))

    roc_after["Ensemble"] = (ens_fpr, ens_tpr, ens_auc)
    proba_after["Ensemble"] = ens_probs
    acc_all["Ensemble\n(after)"] = ens_acc

    plot_roc_multi(roc_after,
                   "ROC Curves — ML Models After SMOTE + Ensemble",
                   f"{MODEL_DIR}/roc_after_smote.png")

    plot_decision_curve(y_test, proba_after,
                        "Decision Curve — After SMOTE + Ensemble",
                        f"{MODEL_DIR}/dca_after_smote.png")

    # ── Step 7 : BERT DL Model ─────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7 — Deep Learning (BERT)")
    print("="*60)
    tokenizer_bert = BertTokenizer.from_pretrained(BERT_NAME)

    def make_loader(texts, labels, shuffle=False):
        ds = DepressionDataset(texts, labels, tokenizer_bert, BERT_MAX_LEN)
        return DataLoader(ds, batch_size=BERT_BATCH, shuffle=shuffle)

    train_loader = make_loader(X_train_raw, y_train, shuffle=True)
    val_loader   = make_loader(X_val_raw,   y_val)
    test_loader  = make_loader(X_test_raw,  y_test)

    dl_models    = get_dl_models()
    roc_dl       = {}
    proba_dl     = {}

    for name, model in dl_models.items():
        print(f"\n{'─'*40}")
        print(f"Training {name}…")
        history = train_dl_model(model, train_loader, val_loader,
                                 epochs=BERT_EPOCHS, lr=BERT_LR)

        plot_dl_history(history, name,
                        f"{MODEL_DIR}/{name.replace('+','_')}_history.png")

        y_true, y_pred, y_prob = evaluate_dl_model(model, test_loader)
        acc     = accuracy_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)

        print(f"  Test Accuracy : {acc:.4f}   AUC : {auc_val:.4f}")
        print(classification_report(y_true, y_pred,
              target_names=["Not Depressed","Depressed"]))

        roc_dl[name]   = (fpr, tpr, auc_val)
        proba_dl[name] = y_prob
        acc_all[name]  = acc

        plot_confusion(y_true, y_pred,
                       f"{name} — Confusion Matrix",
                       f"{MODEL_DIR}/{name.replace('+','_')}_cm.png")

        # Save DL model
        save_path = f"{MODEL_DIR}/{name.replace('+','_')}"
        torch.save(model.state_dict(), save_path + ".pt")
        print(f"  Saved → {save_path}.pt")

    plot_roc_multi(roc_dl,
                   "ROC Curves — BERT",
                   f"{MODEL_DIR}/roc_dl_models.png")

    plot_decision_curve(y_test, proba_dl,
                        "Decision Curve — BERT",
                        f"{MODEL_DIR}/dca_dl_models.png")

    # ── Step 8 : Accuracy Comparison ───────────────────────────
    print("\n" + "="*60)
    print("STEP 8 — Accuracy Comparison (All Models)")
    print("="*60)
    plot_accuracy_comparison(acc_all,
                             f"{MODEL_DIR}/accuracy_comparison.png")

    for k, v in sorted(acc_all.items(), key=lambda x: -x[1]):
        print(f"  {k.replace(chr(10), ' '):30s}: {v*100:.2f}%")

    # ── Step 9 : Save ML Models ────────────────────────────────
    print("\n" + "="*60)
    print("STEP 9 — Saving ML Models & Vectorizer")
    print("="*60)
    pickle.dump(vectorizer,
                open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "wb"))
    for name, model in {**ml_models2, "ensemble": ensemble}.items():
        pickle.dump(model,
                    open(f"{MODEL_DIR}/{name}.pkl", "wb"))
        print(f"  Saved → {MODEL_DIR}/{name}.pkl")

    print("\n✅ Training pipeline complete. All models saved to Google Drive.")
