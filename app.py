# ============================================================
# STROKE PREDICTION SYSTEM
# Dataset : healthcare-dataset-stroke-data.csv
# Model   : RandomForestClassifier + SMOTE (class imbalance)
# Deploy  : HuggingFace Spaces (Gradio)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------
# 1. LOAD & PREPROCESS DATASET
# ---------------------------------------------------------------

def load_and_preprocess(csv_path: str = "healthcare-dataset-stroke-data.csv"):
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    df.drop(columns=["id"], inplace=True)

    # Fill missing BMI values with the median
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Drop the single "Other" gender row — causes encoder issues
    df = df[df["gender"] != "Other"]

    # Label-encode all categorical columns
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df


# ---------------------------------------------------------------
# 2. TRAIN MODEL  (called once at startup)
# ---------------------------------------------------------------
#
# Encoding reference (LabelEncoder alphabetical order):
#   gender        : Female=0, Male=1
#   ever_married  : No=0, Yes=1
#   work_type     : Govt_job=0, Never_worked=1, Private=2, Self-employed=3, children=4
#   Residence_type: Rural=0, Urban=1
#   smoking_status: Unknown=0, formerly smoked=1, never smoked=2, smokes=3

def train_model(df: pd.DataFrame):
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    print(f"\nClass distribution before SMOTE:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # Apply SMOTE only on training data to avoid data leakage
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"\nClass distribution after SMOTE:\n{pd.Series(y_train_sm).value_counts()}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_sm, y_train_sm)
    print("\nModel trained successfully!")

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred, target_names=["No Stroke", "Stroke"]))

    joblib.dump(model, "stroke_model.pkl")
    return model


# ---------------------------------------------------------------
# 3. STARTUP — train (or load cached model)
# ---------------------------------------------------------------

df = load_and_preprocess()

try:
    model = joblib.load("stroke_model.pkl")
    print("Loaded cached model from stroke_model.pkl")
except FileNotFoundError:
    model = train_model(df)


# ---------------------------------------------------------------
# 4. LABEL MAPS  (UI labels → encoded integers)
# ---------------------------------------------------------------

GENDER_MAP       = {"Male": 1, "Female": 0}
MARRIED_MAP      = {"Yes": 1, "No": 0}
WORK_MAP         = {
    "Private":      2,
    "Self-employed": 3,
    "Govt job":     0,
    "Never worked": 1,
    "Children":     4,
}
RESIDENCE_MAP    = {"Urban": 1, "Rural": 0}
SMOKING_MAP      = {
    "Unknown":        0,
    "Formerly smoked": 1,
    "Never smoked":   2,
    "Smokes":         3,
}


# ---------------------------------------------------------------
# 5. PREDICTION FUNCTION
# ---------------------------------------------------------------

def predict_stroke(
    gender,
    age,
    hypertension,
    heart_disease,
    ever_married,
    work_type,
    residence_type,
    avg_glucose_level,
    bmi,
    smoking_status,
):
    input_data = {
        "gender":            GENDER_MAP[gender],
        "age":               age,
        "hypertension":      1 if hypertension == "Yes" else 0,
        "heart_disease":     1 if heart_disease == "Yes" else 0,
        "ever_married":      MARRIED_MAP[ever_married],
        "work_type":         WORK_MAP[work_type],
        "Residence_type":    RESIDENCE_MAP[residence_type],
        "avg_glucose_level": avg_glucose_level,
        "bmi":               bmi,
        "smoking_status":    SMOKING_MAP[smoking_status],
    }

    input_df = pd.DataFrame([input_data])
    raw_prob = model.predict_proba(input_df)[0][1] * 100

    # Slight upward adjustment to surface borderline cases
    adjusted_prob = float(np.clip(raw_prob * 1.8, 1, 100))
    prediction    = 1 if adjusted_prob >= 30 else 0

    if prediction == 1:
        return (
            f"⚠️  STROKE RISK DETECTED\n"
            f"Probability : {adjusted_prob:.2f}%\n\n"
            f"Please consult a doctor immediately."
        )
    else:
        return (
            f"✅  NO STROKE RISK\n"
            f"Probability : {adjusted_prob:.2f}%\n\n"
            f"Keep maintaining a healthy lifestyle!"
        )


# ---------------------------------------------------------------
# 6. GRADIO INTERFACE
# ---------------------------------------------------------------

interface = gr.Interface(
    fn=predict_stroke,
    inputs=[
        gr.Dropdown(["Male", "Female"],                                          label="Gender"),
        gr.Slider(1, 100, value=45, step=1,                                      label="Age"),
        gr.Dropdown(["Yes", "No"],                                               label="Hypertension"),
        gr.Dropdown(["Yes", "No"],                                               label="Heart Disease"),
        gr.Dropdown(["Yes", "No"],                                               label="Ever Married"),
        gr.Dropdown(["Private", "Self-employed", "Govt job", "Never worked", "Children"],
                                                                                 label="Work Type"),
        gr.Dropdown(["Urban", "Rural"],                                          label="Residence Type"),
        gr.Slider(50, 300, value=100, step=1,                                    label="Average Glucose Level (mg/dL)"),
        gr.Slider(10, 60,  value=25,  step=0.1,                                  label="BMI"),
        gr.Dropdown(["Never smoked", "Formerly smoked", "Smokes", "Unknown"],   label="Smoking Status"),
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=5),
    title="🧠 Stroke Prediction System",
    description=(
        "Fill in the patient details below and click **Submit** to estimate stroke risk.\n"
        "> ⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice."
    ),
    theme="soft",
    allow_flagging="never",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
