# Fraud Detection using XGBoost and SMOTE

## Project Overview
This project implements a **Fraud Detection system** for financial transactions using **Machine Learning**. The model uses **XGBoost** for classification and **SMOTE** to handle the imbalanced dataset. The goal is to accurately detect fraudulent transactions.

---

## Dataset
- **Source:** Synthetic Financial Transactions Dataset  
- **Columns:**
  - `step` – timestep of transaction  
  - `type` – transaction type (PAYMENT, TRANSFER, etc.)  
  - `amount` – transaction amount  
  - `oldbalanceOrg`, `newbalanceOrig` – origin account balances  
  - `oldbalanceDest`, `newbalanceDest` – destination account balances  
  - `isFraud` – target variable (1 = fraud, 0 = normal)  
  - `isFlaggedFraud` – flag column (mostly 0)  
  - `nameOrig`, `nameDest` – account IDs (not used for ML)  

---

## Key Steps
1. **Data Exploration**
   - Check for missing values
   - Visualize fraud vs normal transactions
2. **Data Preprocessing**
   - Encode categorical features (`type`)
   - Scale numerical features (`Amount`, `Step`, balances)
   - Handle missing values
3. **Handling Imbalance**
   - Apply **SMOTE** to oversample the minority class (`isFraud = 1`)
4. **Model Training**
   - Train **XGBoost classifier** on the balanced dataset
5. **Evaluation**
   - Classification report (Precision, Recall, F1-score)
   - Confusion matrix
   - ROC-AUC score
   - Feature importance visualization
   - ROC Curve plot
6. **Optional Improvements**
   - Hyperparameter tuning
   - Compare with Random Forest or LightGBM
   - Deploy as a Streamlit/Flask web app

---

## Libraries Used
- `pandas`, `numpy` – data manipulation  
- `scikit-learn` – preprocessing, metrics, train-test split  
- `imbalanced-learn` – SMOTE  
- `xgboost` – ML model  
- `matplotlib`, `seaborn` – visualization  

---

## How to Run
1. Open the `.ipynb` notebook in **Google Colab**.  
2. Upload the dataset CSV file.  
3. Run all cells sequentially to reproduce:
   - Data preprocessing
   - Model training
   - Evaluation and plots  

---

## Resume / Portfolio Highlights
- Developed a **Fraud Detection ML system** using **Python, XGBoost, and SMOTE**.  
- Handled **imbalanced dataset** effectively with SMOTE.  
- Achieved **ROC-AUC > 0.98**.  
- Implemented **feature importance and evaluation visualizations**.  
