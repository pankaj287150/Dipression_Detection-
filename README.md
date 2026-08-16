# Ensemble-Based Depression Detection

### Depression Detection using BERT, Deep Learning and Ensemble Learning

An NLP-based machine learning project that detects depression-related patterns in text using a combination of traditional Machine Learning, Deep Learning, BERT, hybrid models, and ensemble learning.

---

## Overview

Depression is difficult to identify from text because people do not always express their feelings directly. Social media posts and mental-health-related text often contain indirect expressions, informal language, abbreviations, and contextual clues that simple keyword-based methods may miss.

This project explores how different Natural Language Processing and Deep Learning techniques can be used to identify depressive patterns from textual data.

The system compares multiple approaches, starting from traditional Machine Learning models and gradually moving toward Deep Learning and Transformer-based models.

The final system combines the predictions of multiple strong models using **Weighted Soft Voting** to produce the final prediction.

The project achieved approximately **96.59% test accuracy** with the final ensemble and ROC-AUC values above **0.98** across the training, validation, and test datasets. :contentReference[oaicite:2]{index=2}

> **Note:** This project is intended for research and supplementary screening purposes. It is not a replacement for professional medical or clinical diagnosis.



# Problem Statement

Traditional depression screening can be difficult because it depends on factors such as accessibility, availability of professionals, stigma, and geographical limitations.

At the same time, people frequently share their thoughts and experiences through social media and online communities. These posts can contain useful linguistic patterns related to emotional and mental health.

However, detecting depression from text is challenging because:

- People may express distress indirectly
- Social media text is often noisy
- Informal language and abbreviations are common
- URLs and HTML content may appear in posts
- Emotional meaning can depend on the surrounding context
- Important information may occur far apart in a sentence or paragraph

The goal of this project is to build a system that can learn these patterns and classify a text sample as:

**Depressed** or **Non-Depressed**


# What I Built
The project follows a complete NLP and Deep Learning pipeline:

```text
Raw Text
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
Feature Extraction
   ↓
Machine Learning / Deep Learning / BERT
   ↓
Hybrid Models
   ↓
Weighted Ensemble
   ↓
Final Prediction

```



#Project Objectives

Build a text-based depression classification system
Compare traditional Machine Learning with Deep Learning and Transformer models
Study the effectiveness of contextual BERT representations
Develop hybrid models using BERT embeddings
Combine multiple models using ensemble learning
Evaluate the models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC


#Dataset

The project uses two text datasets:

Mental Health Dataset
Reddit Dataset

After cleaning and preprocessing, the combined dataset contains 32,790 labeled samples.

Dataset	Total Samples	Depressed	Non-Depressed
Mental Health Dataset	27,972	13,838	14,134
Reddit Dataset	7,650	3,761	3,889
Combined Dataset	32,790	16,257	16,533

The final dataset is almost balanced:

Depressed: 49.58%
Non-Depressed: 50.42%
Dataset Split
Split	Percentage
Training	70%
Validation	15%
Testing	15%

The training set is used for learning, the validation set for tuning, and the test set for final evaluation.

Dataset

Dataset Link:

https://drive.google.com/drive/folders/149hQleyqGIKaRiOUryzwNj0DncVUScq7?usp=drive_link

#Data Preprocessing

Social media text contains a lot of unnecessary information, so the raw data is cleaned before being passed to the models.

The preprocessing pipeline includes:

1. URL and HTML Removal

URLs, hyperlinks, and HTML tags are removed using regular expressions and BeautifulSoup.

2. Stopword Removal

Common English words such as is, the, and and are removed using NLTK.

3. Tokenization

Text is converted into tokens using NLTK and Hugging Face Transformers tokenizer utilities.

4. Text Normalization

The text is normalized by:

Converting text to lowercase
Removing special symbols
Removing unnecessary spaces

This helps reduce noise and provides cleaner input to the models.

##Models Used

One of the main goals of this project was to compare different types of models.

Machine Learning Models

For traditional Machine Learning, text was converted into numerical features using TF-IDF.

The following models were trained:

Logistic Regression
Support Vector Machine (SVM)
Random Forest

##Deep Learning Models

The text was tokenized and converted into fixed-length sequences.

The following neural network architectures were implemented:

Simple RNN
LSTM
BiLSTM
Bidirectional RNN

These models were used to capture sequential dependencies and contextual patterns in the text.

##Transformer Model
BERT

A pre-trained BERT Base model was fine-tuned on the combined mental-health dataset.

The [CLS] representation from BERT is used as the representation of the complete input text, followed by a classification layer.

BERT achieved the strongest individual performance among the evaluated models, with approximately:

96.54% Accuracy
96.53% F1-Score

##Hybrid Models

To explore whether BERT embeddings could work effectively with traditional classifiers, three hybrid approaches were developed:
```text
BERT
 ↓
[CLS] Embedding
 ↓
 ┌───────────────┬────────────────┬
 ↓               ↓                ↓
SVM          Random Forest      XGBoost
```

The implemented hybrid models are:

BERT + SVM
BERT + Random Forest
BERT + XGBoost

These models use BERT for feature representation and classical Machine Learning algorithms for the final classification.

##Ensemble Learning

The final prediction is generated using a Weighted Soft Voting Ensemble.

Four models are combined:

BERT
BiLSTM
LSTM
BERT + XGBoost

Each model contributes to the final probability based on its assigned weight.

Model	Weight
BERT	0.50
BiLSTM	0.20
LSTM	0.10
BERT + XGBoost	0.20

The final ensemble probability is calculated as:
Pensemble =
0.50 × PBERT
+ 0.20 × PBiLSTM
+ 0.10 × PLSTM
+ 0.20 × PBERT_XGBoost

A threshold of 0.50 is used for the final decision:

Probability >= 0.50  →  Depressed
Probability <  0.50  →  Non-Depressed

Weighted soft voting was chosen because it keeps the probability information from each model instead of simply taking a majority vote.

##Model Architecture
```text

Traditional Machine Learning
Text
 ↓
Preprocessing
 ↓
TF-IDF
 ↓
Logistic Regression / SVM / Random Forest
 ↓
Prediction

Deep Learning
Text
 ↓
Tokenization
 ↓
Sequence Padding
 ↓
RNN / LSTM / BiLSTM / Bidirectional RNN
 ↓
Prediction

#Deep Learning

Text
 ↓
Tokenization
 ↓
Sequence Padding
 ↓
RNN / LSTM / BiLSTM / Bidirectional RNN
 ↓
Prediction

#BERT

Text
 ↓
BERT Tokenizer
 ↓
BERT Transformer
 ↓
[CLS] Representation
 ↓
Classification Layer
 ↓
Prediction
```

##Final Ensemble
```text
                   ┌── BERT ──────────────┐
                   │                       │
                   ├── BiLSTM ────────────┤
Input Text ────────┼── LSTM ──────────────┼── Weighted
                   │                       │   Soft Voting
                   └── BERT + XGBoost ────┘
                                           ↓
                                    Final Prediction
```

#Performance

The models were evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Model Comparison
Model	Accuracy	Precision	Recall	F1-Score
Simple RNN	53.91%	82.65%	10.74%	19.01%
LSTM	92.05%	91.97%	92.00%	91.99%
BiLSTM	92.40%	91.15%	93.77%	92.44%
Bidirectional RNN	91.79%	92.48%	90.82%	91.64%
BERT	96.58%	96.48%	96.64%	96.56%
BERT + Random Forest	87.01%	88.86%	84.38%	86.56%
BERT + SVM	90.59%	90.56%	90.45%	90.50%
BERT + XGBoost	89.98%	90.07%	89.67%	89.87%
Final Ensemble	96.59%	96.51%	96.65%	96.56%

The final ensemble achieved approximately 96.59% accuracy on the held-out test data.

##ROC-AUC Results

The final ensemble achieved approximately:

Dataset Split	ROC-AUC
Training	0.9992
Validation	0.9890
Testing	0.9881

The relatively close AUC values across the three splits indicate consistent performance and good generalization to unseen text.

##Key Results

A few observations from the experiments:

#Simple RNN

The Simple RNN achieved only 53.91% accuracy, showing that a basic recurrent model struggled to capture the longer contextual patterns present in the text.

#LSTM and BiLSTM

Adding gated recurrent architectures significantly improved performance, with both LSTM and BiLSTM crossing 92% accuracy.

#BERT

BERT performed considerably better than the traditional and recurrent models because its self-attention mechanism can capture relationships between words regardless of their distance in the text.

#Ensemble

The final ensemble achieved 96.59% accuracy, combining the strengths of BERT, BiLSTM, LSTM, and BERT-XGBoost.

##Training Setup
The models were trained in GPU-enabled environments using several techniques to improve training stability and generalization.

#Techniques Used
Adam Optimizer
Batch Normalization
Dropout Regularization
Early Stopping
Learning Rate Reduction

Early stopping was used to prevent unnecessary training once validation performance stopped improving, while learning-rate reduction helped the models make finer updates during later stages of training.

##Tech Stack

#Programming & NLP
Python
Natural Language Processing (NLP)
NLTK
Regular Expressions
BeautifulSoup

#Machine Learning

Scikit-learn
Logistic Regression
SVM
Random Forest
TF-IDF

#Deep Learning

TensorFlow / Keras
RNN
LSTM
BiLSTM

#Transformers

BERT
Hugging Face Transformers
Ensemble Learning
Weighted Soft Voting
XGBoost

##Project Structure
```text
Depression-Detection/
│
├── data/
│   ├── mental_health_dataset/
│   └── reddit_dataset/
│
├── preprocessing/
│   ├── data_cleaning.py
│   ├── tokenization.py
│   └── feature_extraction.py
│
├── models/
│   ├── machine_learning/
│   ├── deep_learning/
│   ├── bert/
│   └── ensemble/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── ML_Models.ipynb
│   ├── DeepLearning_Models.ipynb
│   └── BERT_Ensemble.ipynb
│
├── results/
│   ├── confusion_matrix/
│   ├── roc_curves/
│   └── performance_metrics/
│
├── requirements.txt
└── README.md
```
##Applications

This project can be explored for:

Mental health screening support
Social media mental-health analysis
Early depression screening research
NLP-based healthcare research
Mental health research
Intelligent mental-health platforms
Academic and research applications

The system should be treated as a supporting research/screening tool rather than a clinical diagnostic system.

##Future Improvements

There are several directions in which this project can be extended.

Multimodal Depression Detection

Text can be combined with:

Audio signals
Facial expressions
Other behavioural features

This could provide additional information that is not available from text alone.

#Real-Time Application

The model could be deployed as a:

Web application
Mobile application

for accessible and real-time screening support.

Model compression or distillation could also be explored to make BERT-based inference more lightweight.

#Explainable AI

Future versions could use techniques such as:

SHAP
Attention visualization

to make model predictions easier to interpret.

Multilingual Support

The system could be extended to other languages using multilingual Transformer models such as:

mBERT
XLM-R

These improvements could make the system more useful across a wider range of users and datasets.

##Conclusion

This project explores depression detection from text by bringing together several generations of NLP and Machine Learning techniques.

Instead of depending on a single model, the project compares traditional Machine Learning, recurrent neural networks, BERT, hybrid architectures, and finally a weighted ensemble.

The experiments show a clear improvement as the models become better at understanding context. BERT achieved the strongest individual performance, while the final ensemble achieved approximately 96.59% test accuracy with ROC-AUC values above 0.98 across all data splits.

Overall, the project demonstrates how combining different modelling approaches can improve the reliability and generalization of text-based depression classification.

##Disclaimer

This project is developed for educational and research purposes.

The predictions generated by the system should not be considered a medical diagnosis or used as a substitute for evaluation by a qualified mental-health professional.
