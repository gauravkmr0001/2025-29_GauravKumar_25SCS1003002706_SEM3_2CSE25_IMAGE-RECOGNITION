"""
Image Recognition System - Flask Web Application
==================================================
A web app that lets a user upload an image, processes it with OpenCV,
classifies it with a pretrained TensorFlow/Keras (MobileNetV2) model,
and returns the predicted object with a confidence percentage.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
import sqlite3
import uuid
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, g
from werkzeug.utils import secure_filename

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
DATABASE = os.path.join(BASE_DIR, "predictions.db")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB max upload size
IMG_SIZE = 224  # MobileNetV2 expected input size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# --------------------------------------------------------------------------
# Database helpers (SQLite) - stores a history of predictions
# --------------------------------------------------------------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DATABASE) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.commit()


def log_prediction(filename, predicted_class, confidence):
    db = get_db()
    db.execute(
        "INSERT INTO predictions (filename, predicted_class, confidence, created_at) "
        "VALUES (?, ?, ?, ?)",
        (filename, predicted_class, confidence, datetime.utcnow().isoformat()),
    )
    db.commit()


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
# We use MobileNetV2 pretrained on ImageNet (1000 general object classes).
# On first run, Keras downloads the weights automatically (needs internet).
# We then save a local copy to model/model.h5 so the "model" folder in the
# project structure is populated and future loads can use it directly.
print("Loading image classification model (MobileNetV2, ImageNet weights)...")

if os.path.exists(MODEL_PATH):
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    print(f"Loaded cached model from {MODEL_PATH}")
else:
    model = MobileNetV2(weights="imagenet")
    try:
        model.save(MODEL_PATH)
        print(f"Model downloaded and cached at {MODEL_PATH}")
    except Exception as e:  # pragma: no cover
        print(f"Could not cache model to disk ({e}); continuing without cache.")

print("Model ready.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(file_storage):
    """
    Reads an uploaded image using OpenCV and prepares it for MobileNetV2:
    decode -> resize -> BGR to RGB -> normalize via preprocess_input.
    """
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR image

    if img is None:
        raise ValueError("File could not be read as a valid image.")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.expand_dims(img_rgb.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def humanize_label(label):
    return label.replace("_", " ").title()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return (
            jsonify(
                {"error": "Invalid file type. Only JPG, JPEG and PNG are allowed."}
            ),
            400,
        )

    try:
        # Save the uploaded file with a unique, safe filename
        ext = file.filename.rsplit(".", 1)[1].lower()
        safe_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

        file.stream.seek(0)
        raw_bytes = file.read()
        with open(save_path, "wb") as f:
            f.write(raw_bytes)

        # Re-open the saved bytes for OpenCV processing
        file_bytes = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            os.remove(save_path)
            return jsonify({"error": "Uploaded file is not a valid image."}), 400

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb.astype(np.float32), axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=3)[0]  # [(class_id, label, prob), ...]

        top_label = humanize_label(decoded[0][1])
        top_confidence = round(float(decoded[0][2]) * 100, 2)

        other_predictions = [
            {"label": humanize_label(label), "confidence": round(float(prob) * 100, 2)}
            for (_, label, prob) in decoded[1:]
        ]

        log_prediction(safe_name, top_label, top_confidence)

        return jsonify(
            {
                "success": True,
                "prediction": top_label,
                "confidence": top_confidence,
                "other_predictions": other_predictions,
                "image_url": f"/uploads/{safe_name}",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Something went wrong while processing the image: {e}"}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory

    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/history")
def history():
    """Returns the last 20 predictions logged in the SQLite database."""
    db = get_db()
    cur = db.execute(
        "SELECT filename, predicted_class, confidence, created_at "
        "FROM predictions ORDER BY id DESC LIMIT 20"
    )
    rows = cur.fetchall()
    result = [
        {
            "filename": r[0],
            "predicted_class": r[1],
            "confidence": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]
    return jsonify(result)


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File is too large. Maximum size is 8 MB."}), 413


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
