"""
Self-contained script to train the Bayesian Network on Google Colab and export
files the backend can consume:
  - bn_model.pkl       (the trained model + metadata)
  - predictions.csv    (probabilities for every encounter in the dataset)
  - features.json      (the feature list and mutual-information scores)

Upload diabetic_data.csv to your Colab session, then run:
    !python colab_export.py
After it finishes, download the three files from the Files panel and place them
in backend/model/ in your local project.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

CSV_PATH = Path("diabetic_data.csv")
OUT_DIR = Path("colab_export")
OUT_DIR.mkdir(exist_ok=True)

PREDICTIONS_PATH = OUT_DIR / "predictions.csv"
MODEL_PATH = OUT_DIR / "bn_model.pkl"
FEATURES_PATH = OUT_DIR / "features.json"

ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped",
}
DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    16: "Discharged/transferred/referred another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home. Medicaid only, hospice.",
    20: "Expired in a medical facility. Medicaid only, hospice.",
    21: "Expired, place unknown. Medicaid only, hospice.",
    22: "Discharged/transferred to another rehab fac including rehab units of a hospital",
    23: "Discharged/transferred to a long term care hospital.",
    24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility.",
    28: "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
    29: "Discharged/transferred to a Critical Access Hospital (CAH).",
    30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
}
ADMISSION_SOURCE_MAP = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available", 10: "Transfer from critial access hospital",
    11: "Normal Delivery", 12: "Premature Delivery", 13: "Sick Baby",
    14: "Extramural Birth", 15: "Not Available", 17: "NULL",
    18: "Transfer From Another Home Health Agency", 19: "Readmission to Same Home Health Agency",
    20: "Not Mapped", 21: "Unknown/Invalid",
    22: "Transfer from hospital inpt/same fac reslt in a sep claim",
    23: "Born inside this hospital", 24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
}


def classify_diag(code):
    if code == "?":
        return "Missing"
    try:
        c = float(code)
        if 390 <= c <= 459 or c == 785:
            return "Circulatory"
        elif 460 <= c <= 519 or c == 786:
            return "Respiratory"
        elif 520 <= c <= 579 or c == 787:
            return "Digestive"
        elif str(code).startswith("250"):
            return "Diabetes"
        elif 800 <= c <= 999:
            return "Injury"
        elif 710 <= c <= 739:
            return "Musculoskeletal"
        elif 580 <= c <= 629 or c == 788:
            return "Genitourinary"
        elif 140 <= c <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        return "Other"


def has_diabetes(code):
    return 1 if str(code).startswith("250") else 0


def encode_insurance(code):
    if code in ("SP", "FR"):
        return 0
    if code == "?":
        return np.nan
    return 1


def clean_dataframe(df):
    df = df.copy()
    df["admission_type"] = df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Unknown/Invalid")
    df["discharge_disposition"] = df["discharge_disposition_id"].map(DISCHARGE_DISPOSITION_MAP).fillna("Unknown/Invalid")
    df["admission_source"] = df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("Unknown/Invalid")
    df.drop(columns=["admission_type_id", "discharge_disposition_id", "admission_source_id"], inplace=True)
    df.drop(columns=["examide", "citoglipton"], inplace=True, errors="ignore")
    df["diag_1_cat"] = df["diag_1"].apply(classify_diag)
    df["diag_2_cat"] = df["diag_2"].apply(classify_diag)
    df["diag_3_cat"] = df["diag_3"].apply(classify_diag)
    df["diabetes_in_diag1"] = df["diag_1"].apply(has_diabetes)
    df["diabetes_in_diag2"] = df["diag_2"].apply(has_diabetes)

    try:
        from sklearn.impute import KNNImputer
        insurance_to_impute = df["payer_code"].apply(encode_insurance)
        similarity_features = [
            "time_in_hospital", "num_lab_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
        ]
        impute_matrix = df[similarity_features].copy()
        impute_matrix["insurance"] = insurance_to_impute
        imputed = KNNImputer(n_neighbors=5).fit_transform(impute_matrix)
        df["has_insurance"] = np.round(imputed[:, -1]).astype(int)
    except Exception as e:
        print(f"KNN impute failed ({e}); defaulting has_insurance to 1")
        df["has_insurance"] = 1

    df["race"] = df["race"].replace("?", "Unknown")
    df["medical_specialty"] = df["medical_specialty"].replace("?", "Unknown")
    df.drop(columns=["weight"], inplace=True, errors="ignore")
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Not Tested")
    df["A1Cresult"] = df["A1Cresult"].fillna("Not Tested")
    df.drop(columns=["diag_1", "diag_2", "diag_3", "payer_code", "diag_1_cat", "diag_2_cat", "diag_3_cat", "diabetes_in_diag1", "diabetes_in_diag2"], inplace=True, errors="ignore")
    return df


def compute_mutual_information(df, target_col="readmitted", exclude_cols=None, random_state=42):
    exclude_cols = exclude_cols or ["encounter_id", "patient_nbr"]
    exclude = set(exclude_cols + [target_col])
    feature_cols = [c for c in df.columns if c not in exclude]
    X_encoded = pd.DataFrame(index=df.index)
    discrete_mask = []
    for col in feature_cols:
        series = df[col]
        if series.dtype == "object" or str(series.dtype) == "category":
            X_encoded[col] = LabelEncoder().fit_transform(series.astype(str))
            discrete_mask.append(True)
        elif series.nunique() <= 20:
            X_encoded[col] = series
            discrete_mask.append(True)
        else:
            X_encoded[col] = series
            discrete_mask.append(False)
    y = LabelEncoder().fit_transform(df[target_col].astype(str))
    mi = mutual_info_classif(X_encoded, y, discrete_features=discrete_mask, random_state=random_state)
    return pd.DataFrame({"feature": feature_cols, "mutual_information": mi}).sort_values("mutual_information", ascending=False).reset_index(drop=True)


WIDE_NUMERIC_COLS = ["num_lab_procedures", "num_medications", "time_in_hospital"]
CAP_RULES = {"medical_specialty": 15, "discharge_disposition": 10}


def cap_categories(series, top_n=10, other_label="Other"):
    top = series.value_counts().head(top_n).index
    return series.where(series.isin(top), other_label).astype(str)


def bin_numeric(series, labels=None):
    labels = labels or ["Q1_low", "Q2_mid-low", "Q3_mid-high", "Q4_high"]
    return pd.qcut(series.rank(method="first"), q=4, labels=labels).astype(str)


def prepare_bn_data(df, features, target="readmitted"):
    out = df[features + [target]].copy()
    for col, top_n in CAP_RULES.items():
        if col in out.columns:
            out[col] = cap_categories(out[col], top_n=top_n)
    for col in WIDE_NUMERIC_COLS:
        if col in out.columns:
            out[col] = bin_numeric(out[col])
    for col in features:
        if col not in CAP_RULES and col not in WIDE_NUMERIC_COLS:
            out[col] = out[col].astype(str)
    out[target] = out[target].astype(str)
    return out


def stratified_bn_sample(df, n_samples, random_state=42):
    n = min(n_samples, len(df))
    sample, _ = train_test_split(df, train_size=n, stratify=df["readmitted"], random_state=random_state)
    return sample.reset_index(drop=True)


def _extract_dag_edges(dag):
    if hasattr(dag, "edges"):
        raw = dag.edges() if callable(dag.edges) else dag.edges
    else:
        raw = dag
    return [(str(u), str(v)) for u, v in raw]


def _ensure_str_columns(df):
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype(str)
    return out


def _collect_state_names(df):
    df = _ensure_str_columns(df)
    return {col: sorted(df[col].unique().tolist()) for col in df.columns}


def _fit_bn_parameters(model, train_df, state_names=None):
    train_df = _ensure_str_columns(train_df)
    state_names = state_names or _collect_state_names(train_df)
    try:
        from pgmpy.parameter_estimator import DiscreteMLE
        model.fit(train_df, estimator=DiscreteMLE(state_names=state_names))
        return model
    except ImportError:
        pass
    try:
        from pgmpy.estimators import MaximumLikelihoodEstimator
        model.fit(train_df, estimator=MaximumLikelihoodEstimator, state_names=state_names)
        return model
    except TypeError:
        pass
    from pgmpy.estimators import MaximumLikelihoodEstimator
    mle = MaximumLikelihoodEstimator(model=model, data=train_df, state_names=state_names)
    model.add_cpds(*mle.get_parameters())
    return model


def learn_bn_model(structure_df, fit_df=None, target="readmitted", max_indegree=5):
    from pgmpy.models import DiscreteBayesianNetwork
    structure_df = _ensure_str_columns(structure_df)
    fit_df = _ensure_str_columns(fit_df if fit_df is not None else structure_df)
    state_names = _collect_state_names(fit_df)
    edges = None
    try:
        from pgmpy.causal_discovery import HillClimbSearch
        hc = HillClimbSearch(scoring_method="bic-d", return_type="dag", max_indegree=max_indegree, show_progress=False)
        hc.fit(structure_df)
        edges = _extract_dag_edges(hc.causal_graph_)
    except Exception:
        edges = None
    if edges is None:
        from pgmpy.estimators import BicScore, HillClimbSearch
        hc = HillClimbSearch(structure_df)
        dag = hc.estimate(scoring_method=BicScore(structure_df), max_indegree=max_indegree)
        edges = _extract_dag_edges(dag)
    if not edges:
        raise RuntimeError("Structure learning failed — no edges returned.")
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(structure_df.columns.astype(str).tolist())
    model.add_edges_from(edges)
    _fit_bn_parameters(model, fit_df, state_names=state_names)
    return model


def _model_state_names(model):
    states = {}
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd is not None:
            states[node] = list(cpd.state_names[node])
    return states


def _align_df_to_bn_model(model, df, fallback="Other"):
    out = _ensure_str_columns(df)
    known = _model_state_names(model)
    for col in out.columns:
        if col not in known:
            continue
        valid = set(known[col])
        fb = fallback if fallback in valid else known[col][0]
        out[col] = out[col].where(out[col].isin(valid), fb)
    return out


def predict_proba(model, df, target="readmitted", batch_size=1000):
    df = _align_df_to_bn_model(model, df)
    feature_cols = [c for c in df.columns if c != target]
    labels = sorted(model.get_cpds(target).state_names[target])
    predictions = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        preds = model.predict_probability(batch[feature_cols])
        predictions.append(preds[target].values)
    probs = np.concatenate(predictions, axis=0)
    prob_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in labels], index=df.index)
    return prob_df, labels


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    df = clean_dataframe(df)
    print(f"After cleaning: {len(df.columns)} columns")

    print("Computing mutual information...")
    mi = compute_mutual_information(df)
    top_features = mi.head(8)["feature"].tolist()
    print("Top 8 features:", top_features)
    with open(FEATURES_PATH, "w") as f:
        json.dump({"features": top_features, "mi": mi.head(15).to_dict("records")}, f, indent=2)

    print("Training Bayesian Network...")
    bn_df = prepare_bn_data(df, top_features)
    train_df, test_df = train_test_split(bn_df, test_size=0.2, stratify=bn_df["readmitted"], random_state=42)
    fit_df = stratified_bn_sample(train_df, n_samples=30_000, random_state=42)
    structure_sample = stratified_bn_sample(train_df, n_samples=10_000, random_state=42)
    model = learn_bn_model(structure_sample, fit_df=fit_df, target="readmitted", max_indegree=5)
    print(f"Model has {len(model.edges())} edges")

    print("Evaluating on test sample...")
    from sklearn.metrics import accuracy_score, f1_score
    eval_sample = test_df.sample(n=min(5_000, len(test_df)), random_state=42)
    test_aligned = _align_df_to_bn_model(model, eval_sample)
    y_true = test_aligned["readmitted"].values
    preds = model.predict(test_aligned[[c for c in test_aligned.columns if c != "readmitted"]])
    y_pred = preds["readmitted"].values
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Accuracy: {acc:.4f}, macro F1: {macro_f1:.4f}")

    print("Predicting probabilities for all rows...")
    full_bn_df = prepare_bn_data(df, top_features)
    prob_df, labels = predict_proba(model, full_bn_df, target="readmitted", batch_size=1000)

    result = pd.DataFrame({
        "encounter_id": df["encounter_id"].values,
        "patient_nbr": df["patient_nbr"].values,
    })
    for col in prob_df.columns:
        result[col] = prob_df[col].values

    predicted_class = prob_df.columns[np.argmax(prob_df.values, axis=1)].str.replace("prob_", "")
    result["predicted_class"] = predicted_class
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
    print(f"Saved {PREDICTIONS_PATH}")

    joblib.dump({"model": model, "features": top_features, "accuracy": acc, "macro_f1": macro_f1}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
