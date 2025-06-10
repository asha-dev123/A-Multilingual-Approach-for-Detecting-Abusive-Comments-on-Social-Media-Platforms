# A-Multilingual-Approach-for-Detecting-Abusive-Comments-on-Social-Media-Platforms
# 🌐 A Multilingual Approach for Detecting Abusive Comments on Social Media Platforms

This project focuses on building a multilingual classification pipeline to detect **abusive comments** across 13 Indian languages. It combines traditional ML models and deep learning techniques with pretrained transformers like **MuRIL** to tackle the challenge of **low-resource and code-mixed languages** on social media

---

## 🧠 Core Objectives

- Classify abusive vs. non-abusive comments in 13 Indian languages
- Handle language imbalance and low-resource languages using shared multilingual embeddings
- Evaluate model generalizability using cross-lingual validation

---

## 📊 Models Used

| Model Type       | Features / Input            | Description                              |
|------------------|-----------------------------|------------------------------------------|
| Logistic Regression | TF-IDF                     | Classical ML baseline                    |
| Neural Network   | FastText Embeddings         | Feedforward network on word vectors      |
| BiLSTM           | FastText Embeddings         | Sequence modeling for text classification |
| MuRIL Transformer| Sentence Embeddings         | Pretrained multilingual transformer (BERT-based) |

---

## 🛠️ Tech Stack

- Python 3.x
- Scikit-learn
- TensorFlow / Keras
- HuggingFace Transformers
- FastText, Indic NLP Library
- MuRIL by Google Research India

---

## 📁 Project Structure

```bash
multilingual-abuse-detection/
├── data/                 # Preprocessed or raw dataset splits
├── preprocessing/        # Scripts for cleaning, tokenizing, embedding
├── models/               # ML and DL model training scripts
├── evaluation/           # Metrics, confusion matrix, cross-lingual eval
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── visuals/              # Word clouds, charts, etc.
└── README.md             # Project overview
```

---

## 🧪 Evaluation Metrics

- Accuracy
- Macro F1 Score (used due to class imbalance)
- Confusion Matrices
- Cross-lingual performance evaluation

---

## 📚 Dataset

- Provided dataset contains social media comments labeled as abusive or non-abusive across:
  - Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu, Urdu, English

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train MuRIL model
python models/muril_transformer.py

# Evaluate results
python evaluation/cross_lingual_test.py

