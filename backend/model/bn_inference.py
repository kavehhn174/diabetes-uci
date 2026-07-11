"""
Live inference for the trained Bayesian Network stored in
`cleaned_hill_climb_bn_model.pkl`.

The pickle is a dict bundle produced by
`notebooks/bayesian-network-cleaned-hill-climb.ipynb`:

    {
        "model": <pgmpy DiscreteBayesianNetwork>,
        "preprocessor": <CleanedDiscretePreprocessor>,
        "features": [...12 cleaned column names...],
        "target": "readmitted",
        "best_config": {...},
        "val_metrics": {...},
        "test_metrics": {...},
    }

`preprocessor` is an instance of a custom class that was defined in the
notebook's `__main__` at pickle time, so it can only be unpickled if a class
with the exact same name/attributes is importable. We redefine it here
(verbatim) and register it under `__main__` before calling `joblib.load`.

This module exposes a single entry point, `predict_one(query)`, that takes a
dict of patient attributes (raw dataset-style fields, already-cleaned fields,
or a mix of both) and returns the model's predicted probability distribution
over `readmitted` (`NO`, `<30`, `>30`) computed with pgmpy's exact inference
(variable elimination) over the learned CPDs -- i.e. a real forward pass
through the trained network, not a lookup of a precomputed value.
"""

import sys
import warnings
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_PATH = Path(__file__).resolve().parent / "cleaned_hill_climb_bn_model.pkl"


# ---------------------------------------------------------------------------
# CleanedDiscretePreprocessor -- verbatim copy of the class used to train and
# pickle the model (notebooks/bayesian-network-cleaned-hill-climb.ipynb).
# Needed so joblib/pickle can reconstruct `bundle["preprocessor"]` and so its
# `transform` method is actually usable (not just its __dict__).
# ---------------------------------------------------------------------------
class CleanedDiscretePreprocessor:
    def __init__(self, features, target="readmitted", numeric_bins=4, max_categories=12):
        self.features = list(features)
        self.target = target
        self.numeric_bins = numeric_bins
        self.max_categories = max_categories
        self.numeric_bin_edges_ = {}
        self.numeric_labels_ = {}
        self.category_levels_ = {}
        self.numeric_as_category_ = set()

    def fit(self, df):
        for col in self.features:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > self.numeric_bins:
                values = pd.to_numeric(series, errors="coerce").dropna()
                if values.nunique() > 1:
                    _, edges = pd.qcut(
                        values,
                        q=min(self.numeric_bins, values.nunique()),
                        retbins=True,
                        duplicates="drop",
                    )
                    edges = np.unique(edges)
                    if len(edges) > 2:
                        edges[0] = -np.inf
                        edges[-1] = np.inf
                        self.numeric_bin_edges_[col] = edges
                        self.numeric_labels_[col] = [f"Bin_{i}" for i in range(1, len(edges))]
                        continue
            self.numeric_as_category_.add(col)
            self.category_levels_[col] = self._top_categories(series)
        return self

    def _top_categories(self, series):
        values = series.astype("object").where(series.notna(), "Missing").astype(str)
        top_n = max(self.max_categories - 2, 1)
        return values.value_counts().head(top_n).index.tolist()

    def _transform_feature(self, df, col):
        series = df[col]
        if col in self.numeric_bin_edges_:
            values = pd.to_numeric(series, errors="coerce")
            out = pd.cut(
                values,
                bins=self.numeric_bin_edges_[col],
                labels=self.numeric_labels_[col],
                include_lowest=True,
            ).astype("object")
            return out.where(out.notna(), "Missing").astype(str)

        values = series.astype("object").where(series.notna(), "Missing").astype(str)
        known = set(self.category_levels_[col]) | {"Missing", "Other"}
        return values.where(values.isin(known), "Other").astype(str)

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for col in self.features:
            out[col] = self._transform_feature(df, col)
        out[self.target] = df[self.target].astype("object").where(df[self.target].notna(), "Missing").astype(str)
        return out.reset_index(drop=True)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# Register under __main__ so joblib.load can resolve the pickled class
# reference regardless of how this module itself was imported/run.
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "CleanedDiscretePreprocessor", CleanedDiscretePreprocessor)


# ---------------------------------------------------------------------------
# Raw -> cleaned feature derivation
# Mirrors notebooks/bayesian-network-cleaned-hill-climb.ipynb cells 12/16/18.
# ---------------------------------------------------------------------------
ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "Not Available", 7: "Trauma Center", 8: "Not Available",
}

DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home", 2: "Transferred to another hospital", 3: "Transferred to SNF",
    4: "Transferred to ICF", 5: "Transferred to other inpatient care", 6: "Home with home health service",
    7: "Left against medical advice", 8: "Home under IV provider care", 9: "Admitted as inpatient (same hospital)",
    10: "Neonate transferred for aftercare", 11: "Expired", 12: "Still patient / expected back",
    13: "Hospice / home", 14: "Hospice / medical facility", 15: "Transferred to Medicare swing bed",
    16: "Transferred/referred for outpatient services", 17: "Transferred/referred for outpatient services",
    18: "Not Available", 19: "Expired at home (Medicaid hospice)", 20: "Expired in medical facility (Medicaid hospice)",
    21: "Expired, place unknown (Medicaid hospice)", 22: "Transferred to rehab facility",
    23: "Transferred to long term care hospital", 24: "Transferred to nursing facility (Medicaid only)",
    25: "Not Available", 26: "Not Available", 27: "Transferred to federal health care facility",
    28: "Transferred to psychiatric hospital", 29: "Transferred to critical access hospital",
    30: "Transferred to other health care institution",
}

ADMISSION_SOURCE_MAP = {
    1: "Physician referral", 2: "Clinic referral", 3: "HMO referral", 4: "Transfer from hospital",
    5: "Transfer from SNF", 6: "Transfer from other health facility", 7: "Emergency room",
    8: "Court/law enforcement", 9: "Not Available", 10: "Transfer from critical access hospital",
    11: "Normal delivery", 12: "Premature delivery", 13: "Sick baby", 14: "Extramural birth",
    15: "Not Available", 17: "Not Available", 18: "Transfer from another home health agency",
    19: "Readmission to same home health agency", 20: "Not Available", 21: "Not Available",
    22: "Transfer from hospital inpatient (same facility)", 23: "Born inside this hospital",
    24: "Born outside this hospital", 25: "Transfer from ambulatory surgery center", 26: "Transfer from hospice",
}

AGE_GROUP_MAP = {
    "[0-10)": "<30", "[10-20)": "<30", "[20-30)": "<30",
    "[30-40)": "30-60", "[40-50)": "30-60", "[50-60)": "30-60",
    "[60-70)": ">60", "[70-80)": ">60", "[80-90)": ">60", "[90-100)": ">60",
}


def classify_diag(code):
    """Group an ICD-9-CM code into the clinical categories used by the paper (Table 2)."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return "Missing"
    code = str(code).strip()
    if not code or code == "?":
        return "Missing"
    if code.startswith("250"):
        return "Diabetes"
    if code.startswith(("V", "E", "v", "e")):
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def age_bucket_from_years(age_years):
    """Map a numeric age to the same 10-year bucket string used by the raw dataset."""
    edges = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i, edge in enumerate(edges):
        if age_years < edge:
            lo = 0 if i == 0 else edges[i - 1]
            return f"[{lo}-{edge})"
    return "[90-100)"


def _coerce_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


class ValidationError(ValueError):
    def __init__(self, message, missing=None):
        super().__init__(message)
        self.missing = missing or []


# Raw dataset field names that map 1:1 onto a cleaned feature (no transform
# needed beyond int coercion for the "numeric-as-category" BN nodes learned
# by the preprocessor).
DIRECT_NUMERIC_FEATURES = [
    "number_inpatient", "number_outpatient", "number_emergency",
    "number_diagnoses", "time_in_hospital", "num_medications",
    "num_lab_procedures", "num_procedures",
]


def resolve_feature(name, query):
    """Derive the value of a single cleaned feature column from a raw/mixed query dict."""
    if name == "admission_type":
        if query.get("admission_type"):
            return str(query["admission_type"])
        code = _coerce_int(query.get("admission_type_id"))
        return ADMISSION_TYPE_MAP.get(code, "Not Available")

    if name == "discharge_disposition":
        if query.get("discharge_disposition"):
            return str(query["discharge_disposition"])
        code = _coerce_int(query.get("discharge_disposition_id"))
        return DISCHARGE_DISPOSITION_MAP.get(code, "Not Available")

    if name == "admission_source":
        if query.get("admission_source"):
            return str(query["admission_source"])
        code = _coerce_int(query.get("admission_source_id"))
        return ADMISSION_SOURCE_MAP.get(code, "Not Available")

    if name == "diag_1_group":
        if query.get("diag_1_group"):
            return str(query["diag_1_group"])
        if "diag_1" in query and query.get("diag_1") not in (None, ""):
            return classify_diag(query["diag_1"])
        return "Missing"

    if name == "age_group":
        if query.get("age_group"):
            return str(query["age_group"])
        age = query.get("age")
        if age is not None and age != "":
            if isinstance(age, str) and age in AGE_GROUP_MAP:
                return AGE_GROUP_MAP[age]
            age_num = _coerce_int(age)
            if age_num is not None:
                return AGE_GROUP_MAP[age_bucket_from_years(age_num)]
        raise ValidationError("age_group is required (provide 'age_group', 'age' bucket like '[60-70)', or numeric 'age')", ["age_group"])

    if name == "medical_specialty":
        value = query.get("medical_specialty")
        return str(value) if value not in (None, "", "?") else "Missing"

    if name in DIRECT_NUMERIC_FEATURES:
        if name not in query or query[name] in (None, ""):
            raise ValidationError(f"'{name}' is required", [name])
        num = _coerce_int(query[name])
        if num is None:
            raise ValidationError(f"'{name}' must be numeric", [name])
        return num

    # Unknown/forward-compatible feature: pass the raw value through untouched.
    if name in query:
        return query[name]
    raise ValidationError(f"'{name}' is required", [name])


_bundle_lock = Lock()
_bundle_cache = None


def load_bundle():
    global _bundle_cache
    if _bundle_cache is not None:
        return _bundle_cache
    with _bundle_lock:
        if _bundle_cache is not None:
            return _bundle_cache
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        loaded = joblib.load(MODEL_PATH)
        if not isinstance(loaded, dict) or "model" not in loaded or "preprocessor" not in loaded:
            raise ValueError(
                f"{MODEL_PATH} does not look like a model bundle "
                "(expected a dict with 'model'/'preprocessor'/'features' keys)."
            )
        _bundle_cache = loaded
        return _bundle_cache


def _align_to_model_states(model, df):
    """Extra safety net: map any state pgmpy has never seen to a valid fallback
    so VariableElimination never raises on an out-of-vocabulary category."""
    out = df.copy()
    for col in out.columns:
        cpd = model.get_cpds(col)
        if cpd is None:
            continue
        valid = set(cpd.state_names[col])
        val = out.at[out.index[0], col]
        if val not in valid:
            fallback = "Other" if "Other" in valid else ("Missing" if "Missing" in valid else next(iter(valid)))
            out.at[out.index[0], col] = fallback
    return out


def build_cleaned_row(query):
    """Turn a single raw/mixed patient dict into the model's expected cleaned feature row."""
    bundle = load_bundle()
    features = bundle["features"]
    target = bundle.get("target", "readmitted")

    missing = []
    row = {}
    for name in features:
        try:
            row[name] = resolve_feature(name, query)
        except ValidationError as exc:
            missing.extend(exc.missing)
    if missing:
        raise ValidationError(
            f"Missing required field(s): {', '.join(missing)}", missing
        )
    row[target] = None
    return row


def predict_one(query):
    """Run the trained Bayesian Network on a single patient query.

    `query` is a dict that may contain raw dataset fields (e.g.
    `admission_type_id`, `diag_1`, `age`) and/or already-cleaned fields
    (e.g. `admission_type`, `diag_1_group`, `age_group`), plus the six
    numeric utilization counts. Returns predicted class probabilities
    computed by exact inference over the learned CPDs.
    """
    bundle = load_bundle()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    features = bundle["features"]
    target = bundle.get("target", "readmitted")

    row = build_cleaned_row(query)
    df = pd.DataFrame([row])
    cleaned = preprocessor.transform(df)
    feature_cols = [c for c in cleaned.columns if c != target]
    cleaned = _align_to_model_states(model, cleaned[feature_cols])

    raw_probs = model.predict_probability(cleaned)
    labels = sorted(model.get_cpds(target).state_names[target])

    probabilities = {}
    if target in raw_probs.columns:
        # Some pgmpy versions return a single column of dicts/Series.
        cell = raw_probs.iloc[0][target]
        probabilities = dict(cell) if hasattr(cell, "items") else {}
    else:
        for label in labels:
            col = f"{target}_{label}"
            if col not in raw_probs.columns:
                raise KeyError(f"Expected column '{col}' missing from model output: {list(raw_probs.columns)}")
            probabilities[label] = float(raw_probs.iloc[0][col])

    total = sum(probabilities.values()) or 1.0
    probabilities = {k: v / total for k, v in probabilities.items()}
    predicted_class = max(probabilities, key=probabilities.get)

    prob_lt30 = probabilities.get("<30", 0.0)
    prob_gt30 = probabilities.get(">30", 0.0)
    risk_score = round(0.7 * prob_lt30 + 0.2 * prob_gt30, 4)
    if risk_score < 0.12:
        risk_class, risk_label = "low", "Low risk"
    elif risk_score < 0.22:
        risk_class, risk_label = "medium", "Moderate risk"
    else:
        risk_class, risk_label = "high", "High risk"

    return {
        "input_features": row,
        "predicted_class": predicted_class,
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "risk_score": risk_score,
        "risk_class": risk_class,
        "risk_label": risk_label,
    }


if __name__ == "__main__":
    sample_query = {
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "diag_1": "428",
        "age": "[70-80)",
        "medical_specialty": "Cardiology",
        "number_inpatient": 2,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_diagnoses": 9,
        "time_in_hospital": 4,
        "num_medications": 18,
    }
    import json
    print(json.dumps(predict_one(sample_query), indent=2))
