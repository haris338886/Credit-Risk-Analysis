# 💳 Credit Risk Prediction using Tree-Based Models

## 📌 Overview

This project aims to predict **credit risk** using customer financial data. By using machine learning techniques like **Random Forest**, **Gradient Boosting**, and **XGBoost**, we classify individuals as low or high risk. The goal is to assist financial institutions in making informed lending decisions.

---

## 🔍 Dataset

- **Source**: [UCI / Kaggle Credit Risk Dataset]
- **Description**: Contains demographic and financial data for loan applicants, including income, age, debt ratio, and delinquency records.

---

## 🧹 Dataset Preprocessing

- Handled missing values using **median imputation**
- Removed invalid ages (`age <= 0`)
- Created new features:
  - `DebtRatioPerPerson = DebtRatio / NumberOfDependents`
  - `MonthlyIncomePerPerson = MonthlyIncome / NumberOfDependents`
  - `TotalPastDue = sum of past-due columns`
- Scaled features using `StandardScaler`
- Handled class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique)

---

## 🤖 Model Selection & Rationale

- **Random Forest**: Robust and interpretable with built-in feature importance
- **Gradient Boosting**: Strong sequential learner for complex patterns
- **XGBoost**: Fast, regularized, and often state-of-the-art on tabular data

---

## ⚠️ Challenges Faced

| Challenge             | Solution                        |
|----------------------|----------------------------------|
| Imbalanced data       | SMOTE for oversampling           |
| Missing values        | Median imputation                |
| Model overfitting     | Used tree models with default regularization |

---

## 📈 Results & Visualizations

- All models performed well; **XGBoost had the highest ROC-AUC**
- **ROC Curves** visualized to compare performance
- **Feature importance** plot from Random Forest highlighted key predictors (e.g., income, debt ratio)
- Identified and exported list of **high-risk customers** for further action

---
