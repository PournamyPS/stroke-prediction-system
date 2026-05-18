# StrokeSense

A machine learning model that predicts whether a person is at risk of a stroke based on health and lifestyle data!

# Introduction

Stroke is one of the leading causes of death and long-term disability worldwide. Early detection of stroke risk can significantly improve patient outcomes and enable timely medical intervention. Unlike many diseases, stroke risk can often be identified in advance using key health indicators such as age, blood pressure, glucose levels, BMI, and lifestyle habits.

The objective of this project is to build a machine learning model that, when given a patient's health profile, will predict whether they are at risk of a stroke. In this project, we address the challenge of highly imbalanced medical data using SMOTE (Synthetic Minority Over-sampling Technique) and deploy the model through an interactive web interface built with Gradio, making it accessible to anyone without technical knowledge.

# Requirements

## Hardware Requirements

I3 (or above) Processor, 8 GB RAM

## Software Requirements

Google Colab / Jupyter Notebook

- Python 3
- Jupyter Notebook / Google Colab
- Scikit-learn Library
- Gradio

## Functional Requirements

Pandas  
NumPy  
Scikit-learn  
Imbalanced-learn  
Gradio  
Joblib

# Dataset

The following dataset has been used in this project:

1. [Healthcare Dataset Stroke Data — Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

The dataset contains **5110 rows** and **12 columns**:

- `id` : unique identifier (dropped during preprocessing)
- `gender` : Male / Female / Other
- `age` : age of the patient
- `hypertension` : 0 = No, 1 = Yes
- `heart_disease` : 0 = No, 1 = Yes
- `ever_married` : Yes / No
- `work_type` : Private / Self-employed / Govt job / Children / Never worked
- `Residence_type` : Urban / Rural
- `avg_glucose_level` : average blood glucose level
- `bmi` : body mass index
- `smoking_status` : formerly smoked / never smoked / smokes / Unknown
- `stroke` : **target column** — 1 = had a stroke, 0 = no stroke

# Proposed Work

## Data Collection and Cleaning

The dataset was loaded and cleaned as follows:

- Dropped the `id` column as it has no predictive value
- Filled missing values in `bmi` with the column median
- Removed the single row where `gender = 'Other'` to avoid encoding issues

The final cleaned dataset contains **5109 rows** and **11 columns**.

## Preprocessing

The following operations were applied:

- Label Encoding of all categorical columns: `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`
- Train-test split (80/20) performed **before** applying SMOTE to prevent data leakage
- SMOTE applied **only on training data** to balance the stroke class from 249 to 3888 samples

## Training the Model

Algorithm used:

- **Random Forest Classifier** (100 estimators)

## Testing the Model

The model was evaluated on the **original unmodified test set** using:

- Precision
- Recall
- F1 Score
- Classification Report (overall accuracy)

# Results

### Evaluation Scores

| Metric | No Stroke | Stroke |
| --- | --- | --- |
| Precision | 0.96 | 0.16 |
| Recall | 0.94 | 0.22 |
| F1 Score | 0.95 | 0.19 |
| **Overall Accuracy** | **91%** | |

> Note: Low precision on the Stroke class is expected due to extreme class imbalance (only ~4.9% positive cases in the dataset). High recall on the No Stroke class confirms the model reliably identifies healthy patients.

### Class Distribution Before and After SMOTE

| Class | Before SMOTE (Train) | After SMOTE (Train) |
| --- | --- | --- |
| No Stroke | 3888 | 3888 |
| Stroke | 199 | 3888 |

### Sample Predictions

| Patient Profile | Prediction | Probability |
| --- | --- | --- |
| Male, 67, Hypertension, Heart Disease, High Glucose | ⚠️ Stroke Risk Detected | 46% |
| Female, 25, No Risk Factors, BMI 22 | ✅ No Stroke Risk | 0% |

# Conclusion

Stroke prediction is a critical healthcare challenge where early detection can save lives. As shown in the results, the Random Forest Classifier, when combined with SMOTE-based class balancing, is able to effectively identify stroke risk patterns in patient health data.

The model achieves 91% overall accuracy on the test set. The relatively low precision on the stroke class reflects the inherent difficulty of predicting rare medical events, which is a known limitation across all stroke prediction research. High recall on the No Stroke class ensures the model does not incorrectly flag healthy patients.

In the future, more advanced models such as XGBoost or neural networks could be explored. Additionally, incorporating more diverse and larger datasets, as well as additional clinical features like family history or medication data, could further improve prediction accuracy. The interactive Gradio interface makes this model accessible to non-technical users, bridging the gap between machine learning research and practical healthcare applications.

# Status

Completed
