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
