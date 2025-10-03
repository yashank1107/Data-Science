import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide")

# Load CatBoost pipeline (with preprocessing)
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load("cat_pipeline_best.pkl")   # load pipeline instead of only model
        return pipeline
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {e}")
        return None

pipeline = load_model()

# App Title
st.title("📊 Bank Marketing Term Deposit Prediction")
st.markdown("Upload client data (Excel/CSV) and check whether they will subscribe to a term deposit.")

# File uploader
uploaded_file = st.file_uploader("Upload client data file", type=["xlsx", "csv"])

if uploaded_file is not None and pipeline is not None:
    try:
        # Read data
        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)

        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(new_df.head())

        # Predictions (pipeline handles preprocessing + model)
        proba = pipeline.predict_proba(new_df)[:, 1]
        threshold = 0.3   # custom threshold
        preds = (proba >= threshold).astype(int)

        # Add predictions
        results_df = new_df.copy()
        results_df["Prediction"] = preds
        results_df["Probability"] = proba.round(3)

        st.subheader(f"✅ Predictions (Threshold = {threshold})")
        st.dataframe(results_df)

        # Download predictions
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
