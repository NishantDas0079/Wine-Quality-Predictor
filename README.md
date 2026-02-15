# 🍷 Wine Quality Predictor (Red & White)

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://wine-quality-predictor-1-mipw.onrender.com)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-2.3.3-lightgrey)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)

A machine learning web application that predicts whether a wine (red or white) is **acceptable** (quality ≥ 6) or **not acceptable** (quality ≤ 5) based on its physicochemical properties. The model is trained on the combined UCI Wine Quality datasets and deployed using Flask on Render.

🌐 **Live Demo**: [https://wine-quality-predictor-1-mipw.onrender.com](https://wine-quality-predictor-1-mipw.onrender.com)

*(Note: The free tier may spin down after inactivity – the first request might take a few seconds.)*

---

## 📊 Dataset

The model uses the combined **Red** and **White** Wine Quality datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). It contains **6,497 samples** with 12 input features (11 original + `type`):

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
- **Type** (0 = red, 1 = white)

The target variable is a binary class:
- **Acceptable** (original quality score ≥ 6)
- **Not acceptable** (original quality score ≤ 5)

---

## 🚀 Features

- **Unified model** for both red and white wines.
- User-friendly web interface with a clean, responsive design.
- **Wine type dropdown** to select red or white.
- **Six quick‑fill sample buttons** (three for red, three for white) for easy testing.
- JSON API endpoint (`/predict`) for programmatic access.
- Displays prediction, confidence score, and per‑class probabilities.
- **Interactive dashboard** (`/dashboard`) with Plotly visualisations:
  - Alcohol distribution by quality and wine type
  - Feature correlation heatmap
  - Feature importance from the Random Forest model
  - Quality distribution by wine type
- Handles class imbalance using **SMOTE** during training.
- Deployed online – accessible anywhere.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Gunicorn
- **Machine Learning**: scikit‑learn, imbalanced‑learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript
- **Visualisation**: Plotly
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


# 📊 Interactive Dashboard
The app includes a dedicated dashboard at /dashboard with four interactive Plotly charts:

Alcohol Distribution by Quality and Wine Type – box plots showing how alcohol content differs between acceptable/not acceptable wines for red and white.

Feature Correlation Heatmap – visualises correlations among all physicochemical features.

Feature Importance – displays the most influential features according to the trained Random Forest model.

Quality Distribution by Wine Type – bar chart comparing the proportion of acceptable vs. not acceptable wines for red and white.


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
