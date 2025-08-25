from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from model_utils import id3, predict, save_model, load_model

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "mushrooms.csv")
MODEL_PATH = os.path.join(APP_DIR, "mushroom_model.pkl")

app = Flask(__name__)

# Load dataset for form options
df = pd.read_csv(DATA_PATH)

# ✅ Chỉ chọn 3 feature
FEATURES = ["cap-color", "gill-color", "odor"]
CHOICES = {col: sorted(df[col].astype(str).unique().tolist()) for col in FEATURES}

# Train or load model
if os.path.exists(MODEL_PATH):
    tree = load_model(MODEL_PATH)
else:
    X = df[FEATURES]
    y = df["class"]
    tree = id3(X, y, FEATURES)
    save_model(tree, MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", choices=CHOICES)

@app.route("/predict", methods=["POST"])
def predict_html():
    sample = {}
    for f in FEATURES:
        v = request.form.get(f, None)
        if v is None or v == "":
            v = df[f].mode()[0]   # default nếu không chọn
        sample[f] = v

    label = predict(tree, sample)
    human = "🍄 ĂN ĐƯỢC (edible)" if label == "e" else "☠️ ĐỘC (poisonous)"
    return render_template("result.html", sample=sample, label=label, human=human)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(force=True, silent=True) or {}
    sample = {}
    for f in FEATURES:
        v = payload.get(f, None)
        if v is None:
            v = df[f].mode()[0]
        sample[f] = v

    label = predict(tree, sample)
    human = "edible" if label == "e" else "poisonous"
    return jsonify({"label": label, "meaning": human, "sample": sample})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
