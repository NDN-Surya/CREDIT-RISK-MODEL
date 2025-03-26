# Credit Risk Model

## Overview

This repository contains a project aimed at predicting credit risk using advanced machine learning techniques. The dataset includes various features such as credit score, income, and loan amount to assess the likelihood of default effectively.

---


![Credit Risk Model UI](New%20folder%20(2)/Screenshot%20(95).png)



## Motivation

Credit risk prediction is a crucial aspect of the financial industry. Accurate assessment of creditworthiness:

- Helps financial institutions minimize losses.
- Enables responsible lending decisions.
- Empowers borrowers to improve their credit profiles.

This project leverages data-driven methods to predict credit risk, enhancing decision-making in the financial sector.

---

## Goals

The primary objectives of this project are:

1. **Build a machine learning pipeline** for reliable credit risk prediction.
2. **Identify key features** influencing credit risk.
3. **Provide a scalable and user-friendly model** for financial institutions.

---

## Project Structure

The project is organized into the following main components:

### 1. **Main Files**

- `app/main.py`: A script designed to deploy the model as a Streamlit app for user-friendly credit risk prediction.
- `app/prediction_helper.py`: This file assists in scaling input features to match the model’s requirements.

### 2. **Artifacts**

- The `artifacts/` folder contains pre-trained models, including `modeldata.joblib`, and scaling objects used during predictions.

### 3. **Code**

- The `credit_risk_model.ipynb` notebook contains the end-to-end code for building, training, and evaluating the machine learning models.

---

## Technical Aspects

### Data Cleaning

- Missing values were handled using imputation techniques based on feature distributions and domain knowledge.
- Outliers were identified and capped to ensure robust model training without undue influence from extreme values.

### Exploratory Data Analysis (EDA)

Key insights from the EDA process:

- Features such as `loan_tenure_months`, `delinquent_months`, `total_dpd`, and `credit_utilization` were found to strongly correlate with default risk. Higher values in these columns indicated a higher likelihood of default.
- Some features, like `loan_amount` and `income`, did not individually show strong predictive power. However, combining them to create a new feature, `loan_to_income_ratio (LTI)`, revealed a significant impact on the target variable.
- A new feature, `avg_dpd_per_delinquency`, was engineered to capture the average delay in payments per delinquency event, further enhancing model performance.

### Feature Engineering

1. **Feature Creation**:
   - `loan_to_income_ratio`: Ratio of loan amount to income, capturing a borrower’s debt burden.
   - `avg_dpd_per_delinquency`: Average delay in payments per delinquency event.

2. **Feature Selection**:
   - Weight of Evidence (WoE) and Information Value (IV) were used to identify the most predictive features.
   - Variance Inflation Factor (VIF) analysis was conducted to remove multicollinear features, ensuring stability in model predictions.
   - Features with low predictive power and high multicollinearity were excluded, retaining only those with strong relevance to the target variable.

### Model Training and Selection

1. **Baseline Models**:
   - Logistic Regression: Chosen for its simplicity and interpretability.
   - Random Forest: Evaluated for handling non-linear relationships.
   - XGBoost: Tested for its high accuracy and ability to model complex patterns.

2. **Hyperparameter Tuning**:
   - RandomizedSearchCV was applied to optimize hyperparameters for all models.
   - To address class imbalance in the target variable, undersampling, SMOTE, and SMOTENC techniques were tested.
   - Optuna was employed for further hyperparameter optimization.

3. **Final Model Selection**:
   - Logistic Regression and XGBoost performed similarly in terms of accuracy.
   - Logistic Regression was finalized due to its superior interpretability and feature importance insights, which are critical for the financial industry.

### Evaluation

- Evaluation metrics included:
  - Area Under the Curve (AUC-ROC): Assessed the model’s ability to distinguish between classes.
  - K-order statistics: Provided additional robustness to performance evaluation.
  - Gini Coefficient: Measured the discriminatory power of the model.

### Libraries and Tools

- **Python**: Programming language for all scripts and notebooks.
- **Libraries**:
  - `pandas` and `NumPy`: Data manipulation and analysis.
  - `scikit-learn`: Model building, evaluation, and preprocessing.
  - `XGBoost`: Advanced machine learning model.
  - `joblib`: Model serialization.
  - `Streamlit`: Web app deployment.
  - `Optuna`: Hyperparameter optimization.
  - `imbalanced-learn`: Class imbalance techniques.
  - `seaborn` and `matplotlib`: Visualization for EDA and analysis.
- **Tools**:
  - Jupyter Notebook for development and analysis.
  - Streamlit for deployment.

---

## Setup

### Dependencies Installation

Ensure you have Python installed. Clone the repository and run the following commands to install dependencies:

```bash
# Clone the repository
git clone https://github.com/NDN-Surya/CREDIT-RISK-MODEL.git

# Navigate to the project directory
cd CREDIT-RISK-MODEL

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows
venv\Scripts\activate
# For Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

To start the Streamlit app:

```bash
streamlit run app/main.py
```

---

## Future Enhancements

- Incorporating additional features to improve prediction accuracy.
- Expanding the dataset for better generalization.

---

Feel free to contribute or raise issues for further improvements!

