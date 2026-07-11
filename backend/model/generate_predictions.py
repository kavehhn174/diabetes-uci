"""
Load a model exported from Colab (bn_model.pkl) and regenerate predictions.csv
for the local CSV. Use this when you want to update predictions after replacing
bn_model.pkl with a newly trained model.
"""

import sys
import warnings
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "bn_model.pkl"
CSV_PATH = ROOT / "diabetic_data.csv"
PREDICTIONS_PATH = Path(__file__).resolve().parent / "predictions.csv"
FEATURES_PATH = Path(__file__).resolve().parent / "features.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Train the model in Colab and download it into backend/model/."
        )
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict):
        model = loaded["model"]
        features = loaded.get("features")
    else:
        model = loaded
        features = None
    return model, features


def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Export features from notebook into backend/model/."
        )

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if isinstance(payload.get("features"), list):
            return payload["features"]
        if isinstance(payload.get("selected_features"), list):
            return payload["selected_features"]

    raise ValueError(
        f"{FEATURES_PATH} must contain either 'features' or 'selected_features' list."
    )


def clean_for_bn(df, features):
    # Reuse the exact training preprocessors from the notebook export.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from colab_export import clean_dataframe, prepare_bn_data
    sys.path.pop(0)
    cleaned = clean_dataframe(df)
    return prepare_bn_data(cleaned, features)


def generate_predictions():
    model, model_features = load_model()
    file_features = load_features()
    features = model_features or file_features
    if model_features and model_features != file_features:
        print("Warning: features in model bundle differ from features.json; using model bundle features.")

    print(f"Loaded model with features: {features}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} rows")

    bn_df = clean_for_bn(df, features)

    labels = sorted(model.get_cpds("readmitted").state_names["readmitted"])
    predictions = []
    batch_size = 1000

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from colab_export import _align_df_to_bn_model
    sys.path.pop(0)

    aligned = _align_df_to_bn_model(model, bn_df)
    feature_cols = [c for c in aligned.columns if c != "readmitted"]

    for start in range(0, len(aligned), batch_size):
        batch = aligned.iloc[start:start + batch_size]
        preds = model.predict_probability(batch[feature_cols])
        if "readmitted" in preds:
            batch_probs = preds["readmitted"].values
        else:
            expected_cols = [f"readmitted_{label}" for label in labels]
            missing = [col for col in expected_cols if col not in preds.columns]
            if missing:
                raise KeyError(
                    f"Model output is missing expected columns: {missing}. Got: {list(preds.columns)}"
                )
            batch_probs = preds[expected_cols].values
        predictions.append(batch_probs)
        if (start // batch_size) % 10 == 0:
            print(f"  Predicted {min(start + batch_size, len(aligned)):,} / {len(aligned):,}")

    probs = np.concatenate(predictions, axis=0)
    prob_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in labels])
    prob_df = prob_df.apply(pd.to_numeric, errors="coerce")
    if prob_df.isna().any().any():
        fallback = prob_df.mean(numeric_only=True)
        prob_df = prob_df.fillna(fallback)

    result = pd.DataFrame({
        "encounter_id": df["encounter_id"].values,
        "patient_nbr": df["patient_nbr"].values,
    })
    for col in prob_df.columns:
        result[col] = prob_df[col].values

    result["predicted_class"] = prob_df.columns[np.argmax(prob_df.values, axis=1)].str.replace("prob_", "")
    result["risk_score"] = result.get("prob_<30", 0) * 0.7 + result.get("prob_>30", 0) * 0.2

    def classify(score):
        if score < 0.12:
            return "low", "Low risk"
        elif score < 0.22:
            return "medium", "Moderate risk"
        else:
            return "high", "High risk"

    classes = result["risk_score"].apply(classify)
    result["risk_class"] = [c[0] for c in classes]
    result["risk_label"] = [c[1] for c in classes]

    result.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    generate_predictions()
