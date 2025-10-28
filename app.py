from flask import Flask, request, render_template, jsonify, redirect, url_for
import joblib
import numpy as np
import os

# --- konfigurasi model file ---
MODEL_FILE = "model_bundle.joblib"

# --- coba load model bundle dengan aman ---
try:
    bundle = joblib.load(MODEL_FILE)
    model = bundle["final_model"]
    feature_cols = bundle["feature_columns"]
    encoders = bundle["encoders"]  # contains '<col>_leobj' keys for LabelEncoder objects
    print("✅ Model berhasil dimuat.")
except Exception as e:
    print(f"⚠️ Gagal memuat model: {e}")
    model = None
    feature_cols, encoders = [], {}

# --- inisialisasi Flask ---
app = Flask(__name__, static_folder="static", template_folder="templates")


# --- ROUTE HALAMAN UTAMA ---
@app.route("/", methods=["GET", "HEAD"])
def index():
    try:
        categorical_cols = [c for c in feature_cols if (c + "_leobj") in encoders]
        categorical_options = {}
        for c in categorical_cols:
            le = encoders.get(c + "_leobj", None)
            if le is not None:
                categorical_options[c] = list(le.classes_)
            else:
                dictmap = encoders.get(c, None)
                categorical_options[c] = list(dictmap.keys()) if isinstance(dictmap, dict) else []
        return render_template(
            "index.html",
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            categorical_options=categorical_options
        )
    except Exception as e:
        return f"<h3>⚠️ Error rendering page:</h3><pre>{e}</pre>", 500


# --- ROUTE API PREDIKSI ---
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model belum dimuat di server"}), 500

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        input_vals = []
        for col in feature_cols:
            if col not in data:
                return jsonify({"error": f"Missing field: {col}"}), 400
            raw_val = data[col]

            if (col + "_leobj") in encoders:
                le = encoders[col + "_leobj"]
                try:
                    transformed = int(le.transform([raw_val])[0])
                except Exception:
                    classes = list(le.classes_)
                    match_idx = next((i for i, cls in enumerate(classes)
                                      if str(cls).lower() == str(raw_val).lower()), None)
                    if match_idx is None:
                        return jsonify({"error": f"Value '{raw_val}' not recognised for {col}. Allowed: {classes}"}), 400
                    transformed = match_idx
                input_vals.append(transformed)
            else:
                try:
                    input_vals.append(float(raw_val))
                except:
                    return jsonify({"error": f"Invalid numeric for {col}: {raw_val}"}), 400

        arr = np.array(input_vals).reshape(1, -1)
        pred = int(model.predict(arr)[0])
        prob = float(model.predict_proba(arr)[0].max()) if hasattr(model, "predict_proba") else None

        return jsonify({"prediction": pred, "probability": prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- ROUTE FALLBACK (jika form HTML tanpa JS) ---
@app.route("/predict", methods=["POST"])
def predict_form():
    return redirect(url_for('index'))


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)