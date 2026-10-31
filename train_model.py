
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# ==================================================
# 1. FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets"
MODEL_DIR = BASE_DIR / "models"

TRAIN_PATH = DATA_DIR / "fraudTrain.csv"
TEST_PATH = DATA_DIR / "fraudTest.csv"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

TARGET = "is_fraud"
MAX_TRAIN_ROWS = 200_000
MAX_EVAL_ROWS = 50_000


def main():
    print("=" * 55)
    print("FINANCIAL FRAUD DETECTION MODEL")
    print("=" * 55)

    # ==================================================
    # 2. CHECK AND LOAD DATASETS
    # ==================================================

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_PATH}"
        )

    print("\nLoading datasets...")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Training dataset: {train.shape}")
    print(f"Testing dataset: {test.shape}")

    for name, df in [("Training", train), ("Testing", test)]:
        if TARGET not in df.columns:
            raise ValueError(
                f"{name} dataset is missing the '{TARGET}' column. "
                f"Available columns: {df.columns.tolist()}"
            )

    # ==================================================
    # 3. CREATE TIME FEATURES
    # ==================================================

    for df in (train, test):
        if "trans_date_trans_time" in df.columns:
            dt = pd.to_datetime(
                df["trans_date_trans_time"],
                errors="coerce",
            )

            df["transaction_hour"] = dt.dt.hour
            df["transaction_day"] = dt.dt.day
            df["transaction_month"] = dt.dt.month

    # ==================================================
    # 4. SELECT FEATURES
    # ==================================================

    numeric_candidates = [
        "amt",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
        "unix_time",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
    ]

    categorical_candidates = [
        "category",
        "gender",
        "state",
    ]

    numeric_features = [
        col for col in numeric_candidates
        if col in train.columns and col in test.columns
    ]

    categorical_features = [
        col for col in categorical_candidates
        if col in train.columns and col in test.columns
    ]

    features = numeric_features + categorical_features

    if not features:
        raise ValueError(
            "No supported feature columns were found. "
            "Please check the dataset column names."
        )

    print("\nSelected features:")
    print(features)

    X_train = train[features].copy()
    X_test = test[features].copy()

    y_train = pd.to_numeric(
        train[TARGET], errors="raise"
    ).astype(int)

    y_test = pd.to_numeric(
        test[TARGET], errors="raise"
    ).astype(int)

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must be 0 or 1.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Testing labels must be 0 or 1.")

    if y_train.nunique() != 2:
        raise ValueError(
            "Training dataset must contain both classes: 0 and 1."
        )

    if y_test.nunique() != 2:
        raise ValueError(
            "Testing dataset must contain both classes: 0 and 1."
        )

    # Ensure numerical columns contain valid numeric values.
    for col in numeric_features:
        X_train[col] = pd.to_numeric(
            X_train[col], errors="coerce"
        )
        X_test[col] = pd.to_numeric(
            X_test[col], errors="coerce"
        )

    # ==================================================
    # 5. SAMPLE TRAINING DATA IF NECESSARY
    # ==================================================

    if len(X_train) > MAX_TRAIN_ROWS:
        print(
            f"\nSampling {MAX_TRAIN_ROWS:,} training rows "
            "to reduce training time..."
        )

        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_TRAIN_ROWS,
            random_state=42,
            stratify=y_train,
        )

    print(f"Rows used for training: {len(X_train):,}")
    print(f"Fraud cases in training: {(y_train == 1).sum():,}")

    # ==================================================
    # 6. PREPROCESSING PIPELINES
    # ==================================================

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    max_categories=50,
                ),
            ),
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # ==================================================
    # 7. BUILD AND TRAIN THE MODEL
    # ==================================================

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    print("Training completed.")

    # ==================================================
    # 8. PREPARE TEST DATA FOR EVALUATION
    # ==================================================

    if len(X_test) > MAX_EVAL_ROWS:
        print(
            f"\nSampling {MAX_EVAL_ROWS:,} test rows "
            "for faster evaluation..."
        )

        X_eval, _, y_eval, _ = train_test_split(
            X_test,
            y_test,
            train_size=MAX_EVAL_ROWS,
            random_state=42,
            stratify=y_test,
        )
    else:
        X_eval = X_test
        y_eval = y_test

    # ==================================================
    # 9. EVALUATE THE MODEL
    # ==================================================

    print("\nEvaluating model...")

    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

    accuracy = accuracy_score(y_eval, predictions)
    precision = precision_score(
        y_eval, predictions, zero_division=0
    )
    recall = recall_score(
        y_eval, predictions, zero_division=0
    )
    f1 = f1_score(
        y_eval, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_eval, probabilities)
    avg_precision = average_precision_score(
        y_eval, probabilities
    )

    print("\n" + "=" * 55)
    print("MODEL EVALUATION RESULTS")
    print("=" * 55)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"Average Precision:  {avg_precision:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_eval,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_eval,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    # ==================================================
    # 10. SAVE MODEL
    # ==================================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        MODEL_PATH,
    )

    print("\n" + "=" * 55)
    print("SUCCESS: MODEL SAVED")
    print(f"Location: {MODEL_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()
