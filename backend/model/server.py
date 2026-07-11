"""
Python microservice that loads cleaned_hill_climb_bn_model.pkl and serves
live predictions computed by exact inference (pgmpy VariableElimination)
over the trained Bayesian Network. The Node.js backend calls this over HTTP
for on-demand predictions (see backend/src/routes/predict.js).

Usage:
    ./venv/bin/python server.py
    # listens on http://127.0.0.1:5001
"""

import warnings

from flask import Flask, jsonify, request

from bn_inference import ValidationError, load_bundle, predict_one

warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    try:
        bundle = load_bundle()
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "features": bundle.get("features"),
            "target": bundle.get("target"),
            "test_metrics": bundle.get("test_metrics"),
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "model_loaded": False, "error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    if query is None and isinstance(payload, dict) and "query" not in payload:
        # Allow POSTing the query fields directly at the top level too.
        query = payload
    if not isinstance(query, dict) or not query:
        return jsonify({"error": "Request body must include a non-empty 'query' object"}), 400

    try:
        result = predict_one(query)
        return jsonify(result)
    except ValidationError as exc:
        return jsonify({"error": str(exc), "missing": exc.missing}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    load_bundle()  # fail fast on startup if the model file is bad/missing
    app.run(host="127.0.0.1", port=5001, debug=False)
