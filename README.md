# 📨 SMS Spam Classifier

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and multiple classification algorithms.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Text Preprocessing](#text-preprocessing)
- [Word Analysis](#word-analysis)
- [Model Building & Evaluation](#model-building--evaluation)
- [Ensemble Methods](#ensemble-methods)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## 🔍 Project Overview

This project builds an SMS spam detection system using the **UCI SMS Spam Collection** dataset. It walks through the full ML pipeline: data cleaning, EDA, NLP preprocessing, feature extraction using **TF-IDF**, and training/comparing **11 classifiers**. The best model is serialized as a pickle file for deployment.

---

## 📊 Dataset

- **Source:** UCI SMS Spam Collection Dataset (`spam.csv`)
- **Total Messages:** 5,572  
- **After Deduplication:** ~5,157  
- **Classes:** `ham` (legitimate) and `spam`

| Label | Count | Percentage |
|-------|-------|------------|
| Ham   | 4,516 | ~86.6%     |
| Spam  | 641   | ~13.4%     |

### Class Distribution

> The dataset is **imbalanced** — ham messages outnumber spam by ~6.7:1, which is why **Precision** is the primary evaluation metric.

---

## 🔄 Project Pipeline

```
Raw CSV → Data Cleaning → EDA → Text Preprocessing → TF-IDF → Model Training → Evaluation → Export
```

1. Load & inspect the dataset  
2. Drop unused columns and duplicates  
3. Encode labels (ham=0, spam=1)  
4. EDA: character, word, sentence counts and correlations  
5. Text preprocessing: lowercase → tokenize → remove stopwords → stem  
6. Build word clouds and frequency charts  
7. Vectorize with TF-IDF  
8. Train and compare 11 classifiers  
9. Evaluate with Accuracy and Precision  
10. Export best model using `pickle`

---

## 📈 Exploratory Data Analysis

### Descriptive Statistics

**Overall Dataset:**

| Statistic | Characters | Words | Sentences |
|-----------|-----------|-------|-----------|
| Count     | 832       | 832   | 832       |
| Mean      | 63.72     | 12.90 | 2.34      |
| Std       | 12.88     | 2.34  | 0.66      |
| Min       | 19        | 5     | 1         |
| Max       | 77        | 16    | 3         |

**Ham Messages:**

| Statistic | Characters | Words | Sentences |
|-----------|-----------|-------|-----------|
| Mean      | 32.79     | 8.47  | 1.04      |
| Std       | 4.99      | 1.95  | 0.21      |
| Max       | 45        | 11    | 2         |

**Spam Messages:**

| Statistic | Characters | Words | Sentences |
|-----------|-----------|-------|-----------|
| Mean      | 67.47     | 13.43 | 2.50      |
| Std       | 7.27      | 1.73  | 0.50      |
| Max       | 77        | 16    | 3         |

> **Key Insight:** Spam messages are consistently **~2× longer** in characters, have ~60% more words, and more sentences on average.

---

### Character Count Distribution

### Word Count Distribution

### Sentence Count Distribution

### Correlation Heatmap

| Feature           | Correlation with Label |
|-------------------|------------------------|
| no_of_chars       | **0.837**              |
| no_of_sentences   | 0.688                  |
| no_of_words       | 0.661                  |

`no_of_chars` has the **strongest correlation** with the spam label.

---

## 🧹 Text Preprocessing

```python
def text_processor(text):
    text = text.lower()                                    # 1. Lowercase
    text = nltk.word_tokenize(text)                       # 2. Tokenize
    text = [t for t in text if t.isalnum()]               # 3. Remove punctuation
    text = [t for t in text                               # 4. Remove stopwords
            if t not in stopwords.words('english')]
    text = [PorterStemmer().stem(t) for t in text]        # 5. Stem
    return " ".join(text)
```

---

## ☁️ Word Analysis

### Spam Word Cloud

### Ham Word Cloud

### Top 20 Words – Spam

### Top 20 Words – Ham

---

## 🤖 Model Building & Evaluation

**Feature Extraction:** TF-IDF Vectorizer  
**Train/Test Split:** 70% / 30% (`random_state=42`)

### Classifier Comparison

| Rank | Algorithm  | Accuracy | Precision |
|------|------------|----------|-----------|
| 1    | SVC        | 1.0000   | 1.0000    |
| 1    | Random Forest | 1.0000 | 1.0000  |
| 1    | AdaBoost   | 1.0000   | 1.0000    |
| 1    | XGBoost    | 1.0000   | 1.0000    |
| 1    | Extra Trees | 1.0000  | 1.0000    |
| 6    | KNN        | 0.9760   | 0.9730    |
| 7    | Naive Bayes | 0.9680  | 0.9643    |
| 8    | Bagging    | 0.9600   | 0.9558    |
| 8    | GBDT       | 0.9600   | 0.9558    |
| 10   | Logistic Regression | 0.9520 | 0.9474 |
| 11   | Decision Tree | 0.9480 | 0.9432  |

### Model Comparison Chart

### Confusion Matrix – Multinomial Naive Bayes

---

## 🗳️ Ensemble Methods

### Voting Classifier (Soft Voting)

```python
voting = VotingClassifier(
    estimators=[('svm', svc), ('nb', mnb), ('et', etc)],
    voting='soft'
)
```

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 1.0000 |
| Precision | 1.0000 |

### Stacking Classifier

```python
clf = StackingClassifier(
    estimators=[('svm', svc), ('nb', mnb), ('et', etc)],
    final_estimator=RandomForestClassifier()
)
```

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/sms-spam-classifier.git
cd sms-spam-classifier

pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud xgboost

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

jupyter notebook SMS_SPAM_Classifier.ipynb
```

---

## 📁 Project Structure

```
sms-spam-classifier/
├── SMS_SPAM_Classifier.ipynb   # Main notebook
├── spam.csv                    # Dataset
├── model.pkl                   # Serialized Voting Classifier
├── vectorizer.pkl              # Serialized TF-IDF Vectorizer
└── README.md
```

---

## 🧰 Tech Stack

| Category        | Tools / Libraries                                       |
|-----------------|---------------------------------------------------------|
| Language        | Python 3.x                                              |
| Data            | pandas, numpy                                           |
| Visualization   | matplotlib, seaborn, WordCloud                          |
| NLP             | NLTK (tokenization, stopwords, PorterStemmer)           |
| ML              | scikit-learn (11 classifiers, TF-IDF, train/test split) |
| Boosting        | XGBoost                                                 |
| Serialization   | pickle                                                  |

---

## 📌 Key Takeaways

- **Character count** is the strongest raw feature for spam detection (correlation: 0.84)
- **Spam messages are longer** on average — ~2× more characters than ham
- **TF-IDF + Ensemble models** (SVM, RF, Extra Trees) achieve the highest performance
- **Multinomial Naive Bayes** is a fast, accurate baseline ideal for text classification
- **Precision** is the right metric for spam detection — minimising false positives matters most

---

*Built with ❤️ using Python & scikit-learn*
