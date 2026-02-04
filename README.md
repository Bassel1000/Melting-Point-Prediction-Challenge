# 🧪 Melting Point Prediction Challenge

This repository contains my solution for the **Kaggle Melting Point Prediction Challenge**, where the goal is to predict the melting point (Tm) of organic molecules using engineered molecular features.

🔗 **Kaggle Competition:**  
https://www.kaggle.com/competitions/melting-point

🔗 **Kaggle Notebook:**  
https://www.kaggle.com/code/basselashraf/melting-point-prediction-challenge

---

## 📌 Problem Overview

Accurately predicting the melting point of organic compounds is an important task in materials science and chemistry.  
In this competition, the objective is to build a regression model that predicts melting points based on **group contribution and molecular descriptors**.

- **Task:** Regression  
- **Target:** Melting Point (Tm)  
- **Evaluation Metric:** Mean Absolute Error (MAE) *(lower is better)*

---

## 🧠 Approach & Pipeline

The solution follows a structured end-to-end machine learning workflow:

### 1️⃣ Data Loading
- Loaded training and test datasets provided by Kaggle
- Performed initial sanity checks and inspection

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyzed feature distributions
- Identified missing values and outliers
- Examined correlations with the target variable

### 3️⃣ Feature Engineering & Preparation
- Handled missing values
- Feature scaling where necessary
- Created additional engineered features to improve model performance

### 4️⃣ Model Training
Multiple regression models were trained and evaluated, including:

- **XGBoost Regressor**
- **LightGBM Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Ridge Regression**

### 5️⃣ Hyperparameter Optimization
- Used **Optuna** for automated hyperparameter tuning
- Optimized models directly against **MAE**

### 6️⃣ Stacking Ensemble
- Combined multiple strong base learners
- Used a meta-model to improve generalization
- Achieved better performance than individual models

### 7️⃣ Advanced Feature Engineering & Deeper Optimization
- Iterative feature refinement
- Further tuning of ensemble components

---

## 📊 Results

- **Metric:** Mean Absolute Error (MAE)
- Stacking ensemble significantly outperformed single models
- Final predictions submitted to Kaggle leaderboard

*(Exact score may vary depending on the notebook version.)*

---

## 🛠️ Technologies Used

- **Python**
- **NumPy / Pandas**
- **Scikit-learn**
- **XGBoost**
- **LightGBM**
- **Optuna**
- **Matplotlib / Seaborn**

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Bassel1000/melting-point-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost lightgbm optuna matplotlib seaborn
   ```
3. Open the notebook:
   ```bash
   jupyter notebook melting-point-prediction-challenge.ipynb
   ```
