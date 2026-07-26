# 🏦 Bank Marketing Prediction System

A Machine Learning web application that predicts whether a customer will subscribe to a Term Deposit using multiple Machine Learning models.

---

# 📌 Problem Statement

The objective of this project is to predict whether a bank customer will subscribe to a Term Deposit based on customer demographic and banking information.

The application enables users to:

- Upload test data
- Select a Machine Learning model
- Generate predictions
- View model evaluation metrics
- View the confusion matrix
- Download prediction results

---

# 📂 Dataset Description

Dataset: Bank Marketing Dataset

Source: Portuguese Banking Marketing Campaign Dataset

Target Variable

| Variable | Description |
|----------|-------------|
| y | Term Deposit Subscription (Yes / No) |

Input Features

| No | Feature |
|---:|---------|
| 1 | Age |
| 2 | Job |
| 3 | Marital Status |
| 4 | Education |
| 5 | Default |
| 6 | Balance |
| 7 | Housing Loan |
| 8 | Personal Loan |
| 9 | Contact Type |
|10 | Day |
|11 | Month |
|12 | Duration |
|13 | Campaign |
|14 | Pdays |
|15 | Previous |
|16 | Poutcome |

---

# 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree
- K-Nearest Neighbour
- Gaussian Naive Bayes
- Random Forest

---

# 📊 Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|---------:|----:|----------:|--------:|---------:|----:|
| Logistic Regression | 12.23% | 51.26% | 12.08% | 100.00% | 21.56% | 0.0151 |
| Decision Tree | 87.40% | 70.50% | 47.82% | 48.21% | 48.01% | 0.4085 |
| K-Nearest Neighbour | 84.39% | 50.25% | 10.95% | 4.12% | 5.99% | -0.0075 |
| Gaussian Naive Bayes | 82.89% | 81.07% | 34.40% | 46.10% | 39.40% | 0.3012 |
| Random Forest | 89.83% | 90.64% | 67.41% | 30.34% | 41.85% | 0.4072 |

---

# 📈 Performance Summary

| Model | Observation |
|-------|-------------|
| Logistic Regression | Highest recall but very low precision and accuracy. |
| Decision Tree | Good balance between accuracy and recall. Easy to interpret. |
| K-Nearest Neighbour | Good accuracy but poor recall and F1 score. |
| Gaussian Naive Bayes | Moderate performance across all evaluation metrics. |
| Random Forest | Best overall model with the highest Accuracy, AUC, Precision and MCC. Selected as the final model. |

---

# 🌐 Streamlit Application

Features available in the application:

- Upload Test Data
- Select Machine Learning Model
- Generate Predictions
- Display Evaluation Metrics
- Display Confusion Matrix
- Download Prediction Results

---

# 🔗 GitHub Repository

GitHub Repository:

(Add GitHub Repository Link Here)

---

# 🚀 Streamlit Deployment

Streamlit Application:

(Add Streamlit Application URL Here)