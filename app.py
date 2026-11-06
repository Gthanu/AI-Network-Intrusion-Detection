
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Network Intrusion Detection System")

st.markdown("""
### Machine Learning Based Cybersecurity Dashboard

Upload network traffic data to identify potentially malicious activity.
""")

MODEL_PATH = "models/intrusion_model.pkl"

# Check trained model
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run: python train_model.py")
    st.stop()

# Load model
saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

st.sidebar.header("System Information")
st.sidebar.success("AI Model Loaded")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Detection: Binary Classification")

st.subheader("Upload Network Traffic Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Network Data")
    st.dataframe(df.head(20), use_container_width=True)

    missing = [f for f in features if f not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")

    else:
        X = df[features].copy()

        # Validate numeric data
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().any().any():
            st.error("Invalid or missing numeric values found.")
            st.stop()

        # Predict network traffic
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        results = df.copy()

        results["Prediction"] = np.where(
            predictions == 1,
            "Malicious",
            "Normal"
        )

        results["Malicious_Probability"] = (
            probabilities[:, 1] * 100
        ).round(2)

        # Dashboard metrics
        total = len(results)
        malicious = int((predictions == 1).sum())
        normal = total - malicious

        st.subheader("Detection Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Traffic Records", total)
        col2.metric("Normal Traffic", normal)
        col3.metric("Malicious Traffic", malicious)

        st.subheader("Traffic Classification")

        st.bar_chart(
            results["Prediction"].value_counts()
        )

        st.subheader("Detailed Detection Report")

        st.dataframe(
            results,
            use_container_width=True
        )

        # Download results
        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Detection Report",
            data=csv,
            file_name="intrusion_detection_report.csv",
            mime="text/csv"
        )

        if malicious > 0:
            st.error(
                f"Warning: {malicious} records classified as malicious!"
            )
        else:
            st.success("No records classified as malicious.")

else:
    st.info("Upload a compatible network traffic CSV file to begin.")

st.divider()

st.caption(
    "AI Network Intrusion Detection | Python | Scikit-learn | Streamlit"
)
