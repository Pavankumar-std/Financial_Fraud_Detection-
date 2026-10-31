
from pathlib import Path
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Paths and page settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
TEST_PATH = DATA_DIR / "fraudTest.csv"

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Financial Fraud Detection System")
st.caption("Machine Learning | Transaction Risk Analysis")


# --------------------------------------------------
# 2. Load model and data
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data():
    if not TEST_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TEST_PATH)


saved = load_model()

if saved is None:
    st.error(
        "Trained model not found. First run "
        "`python train_model.py` in the project terminal."
    )
    st.stop()

model = saved["model"]
features = saved["features"]

df = load_test_data()

if df.empty:
    st.error(
        "Dataset not found or empty. Check that "
        "Datasets/fraudTest.csv exists."
    )
    st.stop()


# --------------------------------------------------
# 3. Prepare features consistently with training
# --------------------------------------------------

def prepare_features(data):
    data = data.copy()

    if "trans_date_trans_time" in data.columns:
        dt = pd.to_datetime(
            data["trans_date_trans_time"],
            errors="coerce",
        )
        data["transaction_hour"] = dt.dt.hour
        data["transaction_day"] = dt.dt.day
        data["transaction_month"] = dt.dt.month

    for col in features:
        if col not in data.columns:
            data[col] = None

    X = data[features].copy()

    for col in saved["numeric_features"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


# --------------------------------------------------
# 4. Sidebar navigation
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Overview",
        "Model Evaluation",
        "Predict Transactions",
    ],
)


# --------------------------------------------------
# 5. Overview dashboard
# --------------------------------------------------

if page == "Overview":
    st.subheader("Project Overview")

    st.write(
        "This application analyses transaction records and uses "
        "a Random Forest machine learning model to flag transactions "
        "that may require further fraud investigation."
    )

    if "is_fraud" not in df.columns:
        st.error("The test dataset has no is_fraud column.")
        st.stop()

    labels = pd.to_numeric(df["is_fraud"], errors="coerce")
    valid_labels = labels.isin([0, 1])
    labeled_df = df.loc[valid_labels].copy()
    labeled_df["is_fraud"] = labels.loc[valid_labels].astype(int)

    total = len(labeled_df)
    fraud_count = int((labeled_df["is_fraud"] == 1).sum())
    legitimate_count = int((labeled_df["is_fraud"] == 0).sum())
    fraud_rate = (
        fraud_count / total * 100 if total else 0
    )

    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{total:,}")
    b.metric("Fraud Cases", f"{fraud_count:,}")
    c.metric("Legitimate Cases", f"{legitimate_count:,}")
    d.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Transaction Class Distribution")

        distribution = pd.DataFrame({
            "Type": ["Legitimate", "Fraud"],
            "Count": [legitimate_count, fraud_count],
        })

        fig = px.pie(
            distribution,
            names="Type",
            values="Count",
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "category" in labeled_df.columns:
            st.markdown("#### Transaction Categories")

            category_data = (
                labeled_df["category"]
                .fillna("Unknown")
                .value_counts()
                .head(12)
                .rename_axis("Category")
                .reset_index(name="Transactions")
            )

            fig = px.bar(
                category_data,
                x="Category",
                y="Transactions",
            )
            st.plotly_chart(fig, use_container_width=True)

    if "amt" in labeled_df.columns:
        st.markdown("#### Transaction Amounts")

        amount_data = labeled_df[["amt", "is_fraud"]].copy()
        amount_data["Type"] = amount_data["is_fraud"].map({
            0: "Legitimate",
            1: "Fraud",
        })

        # Sample for responsive charts on large datasets.
        if len(amount_data) > 20_000:
            amount_data = amount_data.sample(
                n=20_000, random_state=42
            )

        fig = px.histogram(
            amount_data,
            x="amt",
            color="Type",
            nbins=40,
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 6. Model evaluation
# --------------------------------------------------

elif page == "Model Evaluation":
    st.subheader("Model Performance")

    if "is_fraud" not in df.columns:
        st.error("A labeled test dataset is required.")
        st.stop()

    labels = pd.to_numeric(df["is_fraud"], errors="coerce")
    valid = labels.isin([0, 1])

    evaluation_data = df.loc[valid].copy()
    y = labels.loc[valid].astype(int)

    
    # Use a reproducible, stratified evaluation sample.
    if len(evaluation_data) > 50_000:
        evaluation_data = evaluation_data.copy()
        evaluation_data["_target"] = y

        evaluation_data, _ = train_test_split(
            evaluation_data,
            train_size=50_000,
            random_state=42,
            stratify=evaluation_data["_target"],
        )

        y = evaluation_data["_target"].astype(int)
        evaluation_data = evaluation_data.drop(
            columns=["_target"]
        )

    X = prepare_features(evaluation_data)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Accuracy",
        f"{accuracy_score(y, predictions):.4f}",
    )
    m2.metric(
        "Precision",
        f"{precision_score(y, predictions, zero_division=0):.4f}",
    )
    m3.metric(
        "Recall",
        f"{recall_score(y, predictions, zero_division=0):.4f}",
    )
    m4.metric(
        "F1 Score",
        f"{f1_score(y, predictions, zero_division=0):.4f}",
    )

    st.caption(
        "Metrics are calculated on the labeled test data, using "
        "up to 50,000 sampled rows for dashboard responsiveness."
    )

    cm = confusion_matrix(y, predictions, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"],
    )

    st.markdown("#### Confusion Matrix")

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Count",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    if y.nunique() == 2:
        auc = roc_auc_score(y, probabilities)
        st.metric("ROC-AUC", f"{auc:.4f}")


# --------------------------------------------------
# 7. Predict uploaded transactions
# --------------------------------------------------

elif page == "Predict Transactions":
    st.subheader("Predict Transaction Risk")

    st.write(
        "Upload a CSV containing the same transaction feature "
        "columns as the training dataset. The is_fraud column "
        "is not required."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)

            if new_data.empty:
                st.warning("The uploaded file is empty.")
                st.stop()

            X_new = prepare_features(new_data)

            predictions = model.predict(X_new)
            probabilities = model.predict_proba(X_new)[:, 1]

            results = new_data.copy()

            results["Fraud Prediction"] = [
                "Potential Fraud" if value == 1
                else "Likely Legitimate"
                for value in predictions
            ]

            results["Fraud Risk (%)"] = (
                probabilities * 100
            ).round(2)

            first, second = st.columns(2)

            first.metric("Transactions Analysed", f"{len(results):,}")
            second.metric(
                "Potential Fraud Flags",
                f"{int((predictions == 1).sum()):,}",
            )

            st.dataframe(
                results.head(500),
                use_container_width=True,
            )

            st.download_button(
                label="Download Prediction Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

            st.warning(
                "Predictions indicate model-assessed risk, not "
                "confirmed fraud. Flagged transactions need review."
            )

        except Exception as exc:
            st.error(f"Could not process the uploaded file: {exc}")
            st.info(
                "Check that the CSV has the required transaction "
                "columns and compatible data."
            )
