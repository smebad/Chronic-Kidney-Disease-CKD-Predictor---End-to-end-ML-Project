# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# Loading the scaler and model
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
model_xgb = pickle.load(open('models/xgb_model.pkl', 'rb'))

# features order used in training
FEATURES = ['age', 'bp', 'sg', 'al', 'hemo', 'sc', 'htn', 'dm', 'cad', 'appet', 'pc']
NUMERIC_FEATURES = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
CAT_FEATURES = ['htn', 'dm', 'cad', 'appet', 'pc']

# helpers
@st.cache_data(show_spinner=False)
def load_artifacts():
    with open('models/scaler.pkl', "rb") as f:
        scaler = pickle.load(f)
    with open('models/xgb_model.pkl', "rb") as f:
        model = pickle.load(f)
    return scaler, model

def encode_row(df):
    """Encode categorical fields exactly as training."""
    # Expect original string labels like 'yes'/'no', 'good'/'poor', 'normal'/'abnormal'
    df = df.copy()
    df['htn'] = df['htn'].map({'yes': 1, 'no': 0})
    df['dm']  = df['dm'].map({'yes': 1, 'no': 0})
    df['cad'] = df['cad'].map({'yes': 1, 'no': 0})
    df['appet'] = df['appet'].map({'good': 1, 'poor': 0})
    df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})
    return df

def validate_input_df(df):
    """Ensure all required columns exist."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True

def predict_df(raw_df, scaler, model):
    """Takes raw inputs, encodes, scales and returns predictions+prob."""
    df = raw_df.copy()
    df = encode_row(df)
    # numeric scale
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    # keep order
    df = df[FEATURES]
    preds = model.predict(df)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[:, 1]
    return pd.DataFrame({
        "prediction": preds,
        "probability_ckd": proba if proba is not None else [np.nan]*len(preds)
    }, index=df.index)

# UI for the app
st.set_page_config(page_title="CKD Predictor", layout="wide", page_icon="🩺")
st.title("🩺 Chronic Kidney Disease (CKD) — Predictor")
st.markdown(
    "Use the form below to predict whether a patient may have CKD. "
    "This app uses the model and scaler you trained locally. **This is a demo — do not use for clinical decisions.**"
)

# load artifacts
try:
    scaler, model = load_artifacts()
except Exception as e:
    st.error("Could not load model artifacts. Make sure 'models/scaler.pkl' and 'models/xgb_model.pkl' exist.")
    st.stop()

# twocolumn layout: left = single input, right = quick info + upload
left, right = st.columns([2, 1])

with left:
    st.subheader("Single patient prediction")
    with st.form("input_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0, format="%.0f")
        bp  = c2.number_input("Blood Pressure (bp)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)

        c3, c4 = st.columns(2)
        sg = c3.number_input("Specific gravity (sg)", value=1.02, format="%.2f")
        al = c4.number_input("Albumin (al)", min_value=0, max_value=10, value=1, step=1)

        c5, c6 = st.columns(2)
        hemo = c5.number_input("Hemoglobin (hemo)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
        sc = c6.number_input("Serum creatinine (sc)", min_value=0.0, max_value=50.0, value=1.0, step=0.01)

        htn = st.selectbox("Hypertension (htn)", ["yes", "no"], index=1)
        dm  = st.selectbox("Diabetes (dm)", ["yes", "no"], index=1)
        cad = st.selectbox("Coronary artery disease (cad)", ["yes", "no"], index=1)
        appet = st.selectbox("Appetite (appet)", ["good", "poor"], index=0)
        pc = st.selectbox("Pus cell (pc)", ["normal", "abnormal"], index=0)

        submit = st.form_submit_button("Predict")

    if submit:
        raw = pd.DataFrame([{
            'age': age, 'bp': bp, 'sg': sg, 'al': al, 'hemo': hemo, 'sc': sc,
            'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pc': pc
        }])

        try:
            validate_input_df(raw)
            out = predict_df(raw, scaler, model).iloc[0]
            pred_label = "CKD (positive)" if out['prediction'] == 1 else "Not CKD (negative)"
            prob = out['probability_ckd']
            # display result
            st.markdown("### Prediction")
            if out['prediction'] == 1:
                st.error(f"**{pred_label}**")
            else:
                st.success(f"**{pred_label}**")
            if not np.isnan(prob):
                st.metric(label="Estimated probability of CKD", value=f"{prob:.2%}")
                st.progress(min(max(prob, 0.0), 1.0))
            else:
                st.info("Probability unavailable for this model.")

        except Exception as e:
            st.exception(e)

    st.markdown("---")
    st.info("Tip: use the right panel to upload a CSV with multiple rows for batch predictions.")

with right:
    st.subheader("Batch predictions / Upload CSV")
    st.write("Upload a CSV with columns matching:")
    st.write(FEATURES)
    uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            validate_input_df(df_upload)
            preds_df = predict_df(df_upload, scaler, model)
            out_df = pd.concat([df_upload.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
            st.success(f"Predictions complete — {len(out_df)} rows")
            st.dataframe(out_df.head(20))
            # download button
            csv_bytes = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", csv_bytes, file_name="ckd_predictions.csv", mime="text/csv")
        except Exception as e:
            st.exception(e)

st.markdown("---")
# Feature importance
st.subheader("Model insight")
if hasattr(model, "feature_importances_"):
    try:
        fi = model.feature_importances_
        # ensure length matches FEATURES
        if len(fi) == len(FEATURES):
            fig, ax = plt.subplots(figsize=(8, 4))
            inds = np.argsort(fi)[::-1]
            ax.bar([FEATURES[i] for i in inds], fi[inds])
            ax.set_title("Feature importances (model)")
            ax.set_ylabel("importance")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Model has feature_importances_ but shape doesn't match feature list.")
    except Exception as e:
        st.info("Could not render feature importances.")
else:
    st.info("Feature importances not available for this model.")

st.markdown("""
**Notes & Safety**  
- This app uses a model trained locally. Results are for demonstration only and not clinical advice.
""")
