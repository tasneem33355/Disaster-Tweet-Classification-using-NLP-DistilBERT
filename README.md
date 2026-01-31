# Disaster Tweet Classification using NLP & DistilBERT

## Description
This project focuses on classifying tweets as related to natural disasters or not using **Natural Language Processing (NLP)** and **Machine Learning**. The goal is to help improve disaster response by quickly identifying relevant social media posts.

## Features
- **Data preprocessing:** noise removal, tokenization, lemmatization, stopwords removal
- **Feature extraction:** TF-IDF, additional tweet-based features
- **Models:** Naive Bayes, Logistic Regression, SVM, and **DistilBERT**
- **Evaluation:** Confusion Matrix, Classification Report, Accuracy, Precision, Recall, F1-score
- **Dashboard:** Interactive visualizations using **matplotlib**, **seaborn**, and **Streamlit**
  - Confusion Matrix heatmap
  - Classification Report table
  - Distribution of disaster vs non-disaster tweets
  - Input box to test the DistilBERT model on any tweet

## Results
- Classical ML models reached ~81% accuracy
- DistilBERT achieved **95% accuracy** on the training set and ~83% on unseen validation data

## Usage
1. Load the dataset from Kaggle ([NLP Getting Started](https://www.kaggle.com/c/nlp-getting-started))
2. Preprocess the text data
3. Train the chosen model or use the pre-trained DistilBERT model
4. Visualize results with the interactive dashboard

## Tech Stack
- Python 3.8+
- pandas, numpy
- scikit-learn
- PyTorch & Transformers
- matplotlib, seaborn
- Streamlit
