"""
models.py
=========
Defines:
  • ML models  : Logistic Regression, SVM, Random Forest, Voting Ensemble
  • DL models  : BERT  (PyTorch / HuggingFace)
"""

import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

BERT_NAME = "bert-base-uncased"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# 1.  ML Models (scikit-learn)
# ═══════════════════════════════════════════════════════════════
def get_ml_models():
    """Return (lr, svm, rf, ensemble) — all with probability support."""
    lr  = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                              random_state=42)
    svm = SVC(kernel='rbf', probability=True, C=1.0, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=None,
                                  random_state=42)
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
        voting='soft'
    )
    return lr, svm, rf, ensemble


# ═══════════════════════════════════════════════════════════════
# 2.  Pure BERT Classifier
# ═══════════════════════════════════════════════════════════════
class BertClassifier(nn.Module):
    """
    BERT [CLS] token → Dropout → Linear(768→2)
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained(BERT_NAME)
        self.drop    = nn.Dropout(dropout)
        self.linear  = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs     = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        cls_output  = outputs.pooler_output          # (batch, 768)
        out         = self.drop(cls_output)
        logits      = self.linear(out)               # (batch, 2)
        return logits


# ═══════════════════════════════════════════════════════════════
# 3.  Factory helper
# ═══════════════════════════════════════════════════════════════
def get_dl_models():
    """Return dict with the BERT model instance moved to DEVICE."""
    models = {
        "BERT": BertClassifier(),
    }
    return {name: m.to(DEVICE) for name, m in models.items()}
