# Dipression_Detection-
Ensemble-Based Depression Detection

## Overview
Depression ranks among the most prevalent psychiatric conditions affecting populations worldwide, carrying a significant personal and societal burden. Identifying depressive tendencies at an early stage is vital for enabling timely clinical intervention, providing emotional support, and initiating appropriate treatment pathways.

This project introduces a layered computational framework designed to detect depression from mental health text. The framework integrates classical machine learning, deep learning, pre-trained transformer models, hybrid architectures, and ensemble strategies.

Training and evaluation relied on social media posts and curated mental health corpora containing both depressed and non-depressed examples.

The pipeline was constructed through several distinct phases:

- Raw text cleaning
- TF-IDF-based feature extraction
- Integer sequence tokenization
- Contextual embedding generation via transformer encoders
- Individual model training
- Hybrid classifier development
- Ensemble classifier development

Traditional algorithms such as Logistic Regression, Support Vector Machines, and Random Forest formed the baseline layer. These were complemented by sequential neural architectures including Simple RNN, LSTM, BiLSTM, and Bidirectional RNN.

At the core of the transformer layer, a fine-tuned BERT model was applied for context-sensitive depression classification. Hybrid variants combined BERT-derived embeddings with downstream classifiers such as SVM, Random Forest, and XGBoost.

Experimental findings demonstrate that transformer-derived representations and ensemble fusion consistently outperform purely statistical and shallow recurrent baselines. Among all configurations evaluated, the BERT-based models recorded the highest Accuracy and F1-Score values.

The proposed system may serve as a supplementary screening tool for early depression assessment and as a foundation upon which clinicians and researchers can build more comprehensive intelligent mental health platforms.

# Problem Statement

Mental health disorders have grown into one of the most pressing health challenges of the current era.

Among these, depression stands out for both its prevalence and its far-reaching consequences, ranging from diminished emotional wellbeing and reduced occupational productivity to social isolation and, in severe cases, suicidal ideation.

Catching the condition early substantially improves prognosis, yet conventional screening methods are limited by:

- Clinician availability
- Patient stigma
- Geographical constraints

The proliferation of social media and online mental health communities has generated an unprecedented volume of user-generated text that often reflects genuine psychological states.

This textual evidence, when processed through modern natural language understanding techniques, opens a pathway toward scalable and cost-effective depression screening.

Advances in machine learning, deep learning, and pre-trained language models have demonstrated strong aptitude for classifying emotional and clinical signals embedded in written language.

The present work constructs a comprehensive depression detection pipeline that leverages:

- Classical Machine Learning
- Sequential Neural Networks
- Transformer Encoders
- Hybrid Classifier Combinations
- Multi-model Ensemble Fusion


# Central Challenge

Building an automated system for depression detection is technically demanding because the condition manifests differently across individuals.

The language of depression is rarely explicit. Writers tend to communicate distress through:

- Metaphor
- Minimization
- Indirection

A purely keyword-driven approach therefore captures only a superficial slice of the relevant signal.

Beyond linguistic subtlety, practical challenges arise from data quality.

Real-world social media corpora contain noise in the form of:

- URLs
- Informal abbreviations
- Code-switching
- Inconsistent punctuation

Class imbalance is another recurring problem, since labeled depressive posts frequently constitute a minority of any collected corpus.

Shallow models that cannot learn compositional and long-range syntactic structures struggle to generalize in this setting.

This work addresses these obstacles through:

- Rigorous preprocessing
- Balanced dataset construction
- Deep sequential modeling
- Transfer learning from large-scale pre-trained language representations


# Project Objectives

- Benchmark traditional machine learning classifiers against deep learning and transformer-based alternatives within the same experimental setting
- Design hybrid architectures that pair transformer embeddings with classical downstream classifiers to exploit complementary strengths
- Measure classification quality through a comprehensive suite of metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Quantify the incremental gains offered by ensemble fusion over individual model predictions


# Project Scope

This project is confined to **text-based depression detection** using mental health and social media corpora.

The following modalities fall outside the current scope:

- Audio analysis
- Facial action unit recognition
- Physiological measurement
- Other non-textual modalities

The implemented classification models include:

- BERT-based Transformer Classifier
- BERT + Support Vector Machine (SVM)
- BERT + XGBoost
- BERT + Random Forest
- Ensemble method using Weighted Soft Voting

Performance comparisons also include:

- Logistic Regression
- SVM
- Random Forest
- Simple RNN
- LSTM
- BiLSTM
- Bidirectional RNN

The system is intended as an **assistive screening tool** and is not a substitute for clinical diagnosis.



# Related Work

In recent years, the use of Artificial Intelligence (AI), Machine Learning (ML), and Natural Language Processing (NLP) for depression detection has attracted significant attention.

With the growing popularity of social media platforms such as Twitter and Reddit, people increasingly share their thoughts, emotions, and mental experiences online.

Researchers have explored various approaches including:

- Machine Learning models
- Deep Learning techniques
- Transformer-based architectures
- Hybrid systems
- Multimodal methods

to identify depressive patterns from textual data.

### Classical Machine Learning

Early contributions relied on handcrafted feature engineering combined with classical classifiers such as:

- Naive Bayes
- Support Vector Machines
- Random Forest

These methods extracted:

- Term frequencies
- Sentiment polarities
- Psycholinguistic scores
- LIWC-based features

Their primary limitation was an inability to model:

- Word order
- Contextual ambiguity
- Long-range dependencies

### Recurrent Neural Networks

The introduction of recurrent architectures offered a meaningful step forward.

Models such as:

- LSTM
- Bidirectional LSTM

could encode temporal structure in text, allowing them to capture how meaning evolves across a sentence.

Gated recurrent mechanisms enabled information to persist over longer sequences, which proved beneficial for emotional language where the significance of an early phrase may only become clear at the end of a passage.

### Transformer Architectures

More recently, the field has been transformed by large pre-trained language models built on the Transformer architecture.

BERT and its optimized successors are trained on massive text collections using self-supervised objectives.

When fine-tuned on domain-specific data, these models deliver substantially higher performance on mental health classification tasks compared to both classical and recurrent baselines.

The bidirectional attention mechanism enables the model to weigh every token's relationship with every other token, producing rich contextualized representations.

### Hybrid and Ensemble Learning

Several researchers have explored hybrid and ensemble learning techniques to improve robustness and classification performance.

Hybrid models combining transformer embeddings with classifiers such as:

- SVM
- Random Forest
- XGBoost
- CNN-LSTM

have shown promising results in improving classification accuracy and generalization capability across different datasets.

### Multimodal Depression Detection

Parallel work has explored multimodal depression detection approaches by combining:

- Textual features
- Audio features
- Behavioural features
- Visual features

Researchers have demonstrated that integrating facial expression recognition and speech prosody analysis with text classification improves detection reliability, particularly in clinical interview settings.

The D-Vlog dataset introduced by Yoon et al. exemplifies this multimodal direction. It contains short video blogs annotated for depression, with an accompanying cross-attention framework jointly processing audio and visual streams.

Inspired by these studies, the proposed work focuses on:

- BERT
- Deep Learning models
- Hybrid learning strategies
- Ensemble techniques

for depression detection using social media textual data.


# Existing Approaches

| Author / Year | Method Used | Dataset | Performance | Limitation |
|---|---|---|---:|---|
| Orabi et al. (2018) | BERT fine-tuned on Twitter | Twitter Dataset | 92.10% | Single BERT model; no ensemble fusion; smaller corpus; no hybrid feature combination |
| Wang et al. (2021) | BERT + Handcrafted Features | Social Media | 93.50% | Handcrafted features alongside BERT; no end-to-end fine-tuning; limited model diversity |
| Yadav & Sharma (2022) | BGRU + Textual Analysis | Multi-source | 91.20% | BGRU sequential model only; no transformer fine-tuning; no ensemble fusion strategy |

The comparative study reveals the superiority of transformer and ensemble models over machine learning algorithms in terms of context comprehension and precision.

The suggested framework provides enhanced reliability through the use of:

- Machine Learning
- Deep Learning
- BERT-based models
- Hybrid models
- Ensemble models


# System Architecture

The proposed system follows a multi-stage pipeline consisting of:

1. Dataset collection
2. Data preprocessing
3. Feature extraction
4. Machine Learning models
5. Deep Learning models
6. Transformer-based architecture
7. Hybrid models
8. Ensemble learning
9. Final depression prediction

# Dataset Description

### Dataset Used

The project uses a combination of:

- Reddit Dataset
- Mental Health Dataset

The dataset contains:

- Social media posts
- Mental-health related textual data
- Depression labels
- Non-depression labels

Dataset Link:

https://drive.google.com/drive/folders/149hQleyqGIKaRiOUryzwNj0DncVUScq7?usp=drive_link



# Dataset Contribution and Balance Analysis

Two source datasets were merged to form the final corpus.

After preprocessing and cleaning, the combined dataset yielded a total of **32,790 labeled samples**.

| Dataset | Total Samples | Depressed | Non-Depressed |
|---|---:|---:|---:|
| Mental Health Dataset | 27,972 | 13,838 | 14,134 |
| Reddit Dataset | 7,650 | 3,761 | 3,889 |
| **Combined Dataset** | **32,790** | **16,257** | **16,533** |

### Dataset Labels

| Label | Class |
|---:|---|
| 0 | Non-Depressed |
| 1 | Depressed |

The merged corpus was nearly balanced:

- Depressed: **49.58%**
- Non-Depressed: **50.42%**

The balance ratio was approximately **0.98**, which virtually eliminated the need for synthetic oversampling.

The balanced class distribution improved model generalization capability and ensured fair learning across both depression categories.



# Data Preprocessing

Social media text is inherently messy.

Before any modeling could begin, the raw corpus required systematic cleaning to remove irrelevant content and standardize the remaining text.

### URL and HTML Tag Removal

URLs, hyperlinks, and HTML tags were removed from the textual data to eliminate irrelevant information and noise that do not contribute to depression detection.

Regular expressions (`re`) and the BeautifulSoup library from `bs4` were used for this preprocessing step.

### Stopword Removal

Commonly occurring words such as:

- `is`
- `the`
- `and`

were removed to reduce redundancy and improve feature quality.

The NLTK (Natural Language Toolkit) library was used to remove English stopwords.

### Tokenization

Tokenization is the process of splitting text into smaller units called tokens, such as words or subwords.

This helps machine learning and transformer models process textual information effectively.

Tokenization was performed using:

- NLTK
- Hugging Face Transformers tokenizer utilities

### Text Normalization

Text normalization was applied to convert text into a consistent format by:

- Converting characters into lowercase
- Removing special symbols
- Handling unnecessary spaces

Python libraries such as `re` and NLTK were used for normalization.

The preprocessing stage improved textual consistency and helped the models learn meaningful depression-related semantic patterns.



# Exploratory Data Analysis

EDA was performed to understand textual characteristics and statistical patterns within the dataset.

### Class Distribution Analysis

The class distribution was analyzed for:

- Depressed samples
- Non-Depressed samples

### Word Count Distribution

Word count distribution was analyzed separately for:

- Depressed class
- Non-Depressed class

These analyses helped understand the statistical characteristics of the dataset before model training.


# Data Splitting

The preprocessed dataset was divided into training, validation, and testing subsets using a **70:15:15 ratio**.

| Dataset | Percentage |
|---|---:|
| Training Set | 70% |
| Validation Set | 15% |
| Testing Set | 15% |

The:

- Training set was used for model learning
- Validation set was used for hyperparameter tuning
- Testing set was used for final performance evaluation on unseen data


# Machine Learning Pipeline

The machine learning pipeline utilized **TF-IDF vectorization** to convert textual data into numerical feature representations.

Traditional machine learning algorithms including:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

were then trained to classify depressed and non-depressed textual samples.


# Deep Learning Pipeline

The deep learning pipeline utilized tokenization and padding techniques to convert textual data into fixed-length numerical sequences.

The following architectures were implemented:

- Simple RNN
- LSTM
- BiLSTM
- Bidirectional RNN

These models were used to capture sequential dependencies and contextual emotional patterns for depression detection.


# BERT Embedding Based Classifiers

Hybrid architectures were built by extracting the `[CLS]` token embedding produced by a fine-tuned BERT encoder and routing it to a downstream classifier.

This two-stage design separates:

- Representation learning — handled by the Transformer
- Decision boundary estimation — handled by the classical classifier

The hybrid models developed include:

- BERT + SVM
- BERT + Random Forest
- BERT + XGBoost


# Ensemble Learning

The ensemble layer combines probability outputs from four constituent models:

- BERT
- BiLSTM
- LSTM
- BERT_XGBoost

Weighted soft voting was chosen because it preserves probabilistic information unlike hard voting, which discards confidence scores.

Weights were assigned in proportion to each model's observed validation-set performance.

| Model | Assigned Weight |
|---|---:|
| BERT | 0.50 |
| BiLSTM | 0.20 |
| LSTM | 0.10 |
| BERT_XGBoost | 0.20 |


# Final Prediction Output

After weighted aggregation, the ensemble emits a probability score for the depressed class.

The final ensemble probability is calculated as:


