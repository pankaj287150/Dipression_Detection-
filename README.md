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
