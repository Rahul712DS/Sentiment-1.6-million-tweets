# 🐦 Twitter Sentiment Analysis on 1.6M Tweets

## 📌 Project Overview

This project performs **sentiment analysis on 1.6 million tweets**, classifying them as **positive** or **negative** using Natural Language Processing (NLP) and Machine Learning models.

The workflow covers the **entire ML pipeline**:

* Data loading & preprocessing
* Cleaning and lemmatization
* Feature extraction with TF-IDF
* Model building (Logistic Regression, SVM)
* Model evaluation
* Model saving & loading for predictions

---

## 🎯 Problem Statement

The goal is to build a machine learning model that can classify tweets into **positive** or **negative sentiments**. This can help businesses, organizations, and researchers analyze public opinion at scale.

---

## 📂 Dataset

* **Source**: [Sentiment140 Dataset](http://help.sentiment140.com/for-students)
* **Size**: 1.6 million tweets
* **Columns**:

  * `target`: Sentiment (0 = negative, 4 = positive → mapped to 0 and 1)
  * `ids`: Tweet ID
  * `date`: Date of the tweet
  * `flag`: Query flag
  * `user`: User handle
  * `text`: Tweet text

---

## 🛠️ Tech Stack / Skills

* **Languages**: Python
* **Libraries**: Pandas, NumPy, NLTK, Scikit-learn, Seaborn, Matplotlib, TQDM
* **ML Models**: Logistic Regression, Support Vector Machine (SVM)
* **Vectorization**: TF-IDF
* **Persistence**: Pickle for model saving/loading

---

## 🔑 Key Steps

1. **Data Preprocessing**

   * Removed URLs, mentions, hashtags, non-alphanumeric characters
   * Converted text to lowercase, removed stopwords
   * Lemmatization with POS tagging

2. **Feature Engineering**

   * Applied **TF-IDF Vectorization** (max 5000 features)

3. **Model Training**

   * Logistic Regression (baseline)
   * Support Vector Machine (LinearSVC)

4. **Evaluation**

   * Accuracy used as metric
   * Logistic Regression: 77.17% 
   * SVM: 77.14%

5. **Deployment-Ready**

   * Model and vectorizer saved as `sentiment_model_lr.pkl`
   * Example usage included for making predictions on new text

---

## 📊 Results

* Successfully trained models on **1.6M tweets**
* Demonstrated the feasibility of real-time sentiment classification
* Insights: Negative and Positive classes were balanced after preprocessing

---

## 🚀 How to Run

 Load model & test predictions

```python
import pickle
with open("sentiment_model_lr.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

sample = ["I love this product!", "This is the worst day ever"]
X_new = vectorizer.transform(sample)
print(model.predict(X_new))
```

---

## 📌 Future Improvements

* Use **deep learning (LSTM / BERT)** for better accuracy
* Deploy as a **Streamlit web app**
* Perform **real-time sentiment analysis** via Twitter API

---

## 📎 Deliverables

* Source code: `Sentimental_code.py`
* Preprocessed dataset (pickle)
* Trained models (`sentiment_model_lr.pkl`)
* Documentation (this README)

---

## 👨‍💻 Author

**Rahul Raj**
📍 Jamshedpur, Jharkhand | [LinkedIn](#) | [GitHub](#)

---
