# Image Recognition System

A simple, full-stack web application that lets a user upload an image
(JPG/JPEG/PNG), previews it in the browser, and uses a **pretrained
TensorFlow/Keras deep learning model (MobileNetV2)** together with
**OpenCV** for image preprocessing to identify the main object in the
image, returning the predicted class and a confidence percentage.

Built for a college mini-project using: **Python, Flask, TensorFlow/Keras,
OpenCV, HTML/CSS/JavaScript, and SQLite**.

---

## 1. Features

- Clean, responsive, modern homepage
- Drag-and-drop image upload (plus a normal "Browse" button)
- Client-side and server-side validation for JPG / JPEG / PNG only
- Live image preview before prediction
- Image preprocessing with OpenCV (decode, resize, color conversion)
- Object classification using a pretrained MobileNetV2 CNN (ImageNet, 1000 classes)
- Displays the predicted class + confidence percentage, plus the next
  2 runner-up predictions
- Clear, friendly error messages for invalid files or oversized uploads
- Animated loading spinner while the model is processing the image
- "Upload Another Image" button to reset and try again
- SQLite database that logs every prediction (filename, class, confidence,
  timestamp) for a simple prediction history

---

## 2. Project Structure

```
image_recognition/
├── app.py                 # Flask backend: routes, OpenCV preprocessing, model inference
├── model/
│   └── model.h5            # Cached copy of the pretrained model (created on first run)
├── templates/
│   └── index.html          # Homepage (upload UI, preview, results)
├── static/
│   ├── style.css            # Styling (responsive, modern UI)
│   └── script.js             # Drag & drop, preview, AJAX calls, UI state
├── uploads/                # Uploaded images are stored here
├── predictions.db          # SQLite database (auto-created on first run)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 3. About the Model (Important)

This project uses a **pretrained model instead of training one from
scratch**, which is the standard, practical approach for an image
classification demo/college project (training a competitive CNN from
scratch requires large labeled datasets and significant compute time).

- **Model:** `MobileNetV2` from `tensorflow.keras.applications`
- **Weights:** pretrained on **ImageNet**, which covers **1,000 general
  object categories** (animals, everyday objects, vehicles, food, etc.)
- **How it's loaded:** On the first run, `app.py` downloads the ImageNet
  weights automatically via Keras (`MobileNetV2(weights="imagenet")`)
  and then saves a local copy to `model/model.h5`. On every subsequent
  run, the app loads the model directly from `model/model.h5`, so it
  works fully offline after the first successful run.
- **How it's used for prediction:**
  1. The uploaded image is read and decoded using **OpenCV**
     (`cv2.imdecode`).
  2. It is resized to `224x224` pixels (the input size MobileNetV2
     expects) using `cv2.resize`.
  3. The color order is converted from OpenCV's default BGR to RGB
     using `cv2.cvtColor`.
  4. The array is expanded to a batch of size 1 and normalized with
     Keras's built-in `preprocess_input` for MobileNetV2.
  5. `model.predict()` returns probabilities across the 1,000 ImageNet
     classes.
  6. `decode_predictions()` converts the raw output into human-readable
     labels, and the top-1 label + probability are shown to the user as
     the **prediction** and **confidence percentage** (probability × 100).

> If you want to use your **own custom-trained model** instead, simply
> replace the model-loading section in `app.py` with
> `load_model("model/model.h5")` pointing to your own `.h5` file, and
> update `preprocess_image()` / `decode_predictions` logic to match your
> model's expected input size and class labels.

---

## 4. Installation

### Step 1 — Prerequisites
- Python 3.9 – 3.11 installed
- `pip` package manager

### Step 2 — Get the project files
Place the `image_recognition/` folder anywhere on your machine, then
open a terminal inside it:

```bash
cd image_recognition
```

### Step 3 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow can take a few minutes to install and the first
> app run will download ~14 MB of MobileNetV2 weights, so make sure
> you have an internet connection the first time you run the app.

---

## 5. Running the Application

```bash
python app.py
```

You should see output similar to:

```
Loading image classification model (MobileNetV2, ImageNet weights)...
Model downloaded and cached at .../model/model.h5
Model ready.
 * Running on http://127.0.0.1:5000
```

Now open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 6. How to Use

1. Drag and drop an image onto the upload area, or click **Browse
   Files** to select one (JPG, JPEG or PNG only, max 8 MB).
2. A preview of your image appears.
3. Click **Identify Object**.
4. A loading spinner appears while the model processes the image.
5. The predicted object and confidence percentage are displayed,
   along with two runner-up predictions.
6. Click **Upload Another Image** to reset and try again.
7. If you upload an unsupported file type or an oversized file, a
   clear error message is shown instead of crashing the app.

---

## 7. Prediction History (SQLite)

Every prediction is logged to `predictions.db` (created automatically)
in a table called `predictions` with columns:
`id, filename, predicted_class, confidence, created_at`.

You can view the last 20 predictions as JSON by visiting:

```
http://127.0.0.1:5000/history
```

---

## 8. Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` |
| TensorFlow install fails on an older Python version | Use Python 3.9–3.11 |
| First run is slow / no internet | The first run needs internet to download model weights once; after that it runs offline using `model/model.h5` |
| "File is too large" error | Upload an image smaller than 8 MB, or increase `MAX_CONTENT_LENGTH` in `app.py` |
| Port 5000 already in use | Change the port in the last line of `app.py`, e.g. `app.run(debug=True, port=5001)` |

---

## 9. Tech Stack Summary

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Deep Learning | TensorFlow / Keras (MobileNetV2, pretrained on ImageNet) |
| Image Processing | OpenCV |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Database | SQLite (prediction history log) |

---

Enjoy exploring the Image Recognition System! 🎉
