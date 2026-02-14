# 🍷 Wine Quality Predictor

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://wine-quality-predictor-1-mipw.onrender.com)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-2.3.3-lightgrey)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)

A machine learning web application that predicts whether a red wine is **acceptable** (quality ≥ 6) or **not acceptable** (quality ≤ 5) based on its physicochemical properties. The model is trained on the UCI Wine Quality dataset and deployed using Flask on Render.

🌐 **Live Demo**: [https://wine-quality-predictor-1-mipw.onrender.com](https://wine-quality-predictor-1-mipw.onrender.com)

*(Note: The free tier may spin down after inactivity – the first request might take a few seconds.)*

---

## 📊 Dataset

The model uses the [Red Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from UCI. It contains 1,599 samples with 11 input features:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The target variable is a binary class:
- **Acceptable** (original quality score ≥ 6)
- **Not acceptable** (original quality score ≤ 5)

---

## 🚀 Features

- User-friendly web interface with a clean, responsive design.
- Quick‑fill sample buttons for easy testing.
- JSON API endpoint (`/predict`) for programmatic access.
- Displays prediction and confidence score.
- Handles class imbalance using **SMOTE** during training.
- Deployed online – accessible anywhere.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Gunicorn
- **Machine Learning**: scikit‑learn, imbalanced‑learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render (free tier)

---

## 🧪 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/NishantDas0079/Wine-Quality-Predictor.git
cd Wine-Quality-Predictor
```

# 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

# 3. Install dependencies
```bash
pip install -r requirements.txt
```

# 4. Run the Flask App
```bash
python app.py
```

# Open the browser
```
http://127.0.0.1:5000
```

# 📡 API Usage
You can also interact with the model via a JSON POST request.

Endpoint: `https://wine-quality-predictor-1-mipw.onrender.com/predict`

# 🤖 Model Training
The classifier is a Random Forest trained with SMOTE to balance the classes. Key steps:

Load and preprocess data.

Create binary target (acceptable/not acceptable).

Split into train/test sets.

Build a pipeline: `StandardScaler → SMOTE → RandomForestClassifier`.

Train and save with `joblib`.

See `train_model_binary.py` for details (if included).

## 📈 Performance

The model was evaluated on a held-out test set (20% of the data, stratified by class). Below are the detailed metrics:

| Class           | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| Not acceptable  | 0.80      | 0.79   | 0.79     | 149     |
| Acceptable      | 0.82      | 0.82   | 0.82     | 171     |

| Metric          | Value |
|-----------------|-------|
| **Accuracy**    | 0.81  |
| Macro Avg       | 0.81  |
| Weighted Avg    | 0.81  |

- **Overall accuracy**: **81%** on unseen data.
- The model performs consistently well for both classes, with slightly higher precision and recall for the "acceptable" class.
- The weighted average F1-score of **0.81** indicates good balance between precision and recall across the imbalanced classes (thanks to SMOTE).
