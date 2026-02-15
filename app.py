# app.py (combined model + dashboard)
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np

# Load the combined model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'wine_quality_combined.pkl')
model = joblib.load(model_path)

# Feature names (now including 'type')
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates',
    'alcohol', 'type'
]

app = Flask(__name__)

# ------------------------------------------------------------
# Home page – prediction form
# ------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------------------------------------
# JSON API endpoint (for programmatic access)
# ------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = df[feature_names]
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        class_labels = ['not acceptable', 'acceptable']
        result_label = class_labels[prediction]
        response = {
            'quality': result_label,
            'confidence': round(probabilities[prediction], 3),
            'probabilities': {
                'not acceptable': round(probabilities[0], 3),
                'acceptable': round(probabilities[1], 3)
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# ------------------------------------------------------------
# Form submission – result page
# ------------------------------------------------------------
@app.route('/result', methods=['POST'])
def result():
    try:
        # Extract features from form
        features = {}
        for name in feature_names:
            if name == 'type':
                features[name] = int(request.form[name])
            else:
                features[name] = float(request.form[name])

        df = pd.DataFrame([features])
        df = df[feature_names]
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        class_labels = ['not acceptable', 'acceptable']
        result_label = class_labels[prediction]

        # Get wine type for display
        wine_type = "Red" if features['type'] == 0 else "White"

        # Build probability list HTML
        prob_rows = ""
        for label, prob in zip(class_labels, probabilities):
            prob_rows += f"<li>{label}: <strong>{prob:.3f}</strong></li>"

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .result-box {{ background: #f0f8ff; padding: 20px; border-radius: 8px; max-width: 500px; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ margin: 8px 0; }}
                .acceptable {{ color: green; }}
                .not-acceptable {{ color: red; }}
                .back-link {{ margin-top: 20px; display: inline-block; }}
            </style>
        </head>
        <body>
            <h2>🍷 Wine Quality Predictor</h2>
            <div class="result-box">
                <p><strong>Wine type:</strong> {wine_type}</p>
                <p><strong>Quality:</strong> 
                    <span class="{'acceptable' if prediction == 1 else 'not-acceptable'}">
                        {result_label}
                    </span>
                </p>
                <p><strong>Confidence:</strong> {probabilities[prediction]:.3f}</p>
                <h3>Probabilities:</h3>
                <ul>
                    {prob_rows}
                </ul>
            </div>
            <a href="/" class="back-link">← Predict another wine</a>
        </body>
        </html>
        """
    except Exception as e:
        return f"<h2>Error</h2><p>{str(e)}</p><a href='/'>Go back</a>"

# ------------------------------------------------------------
# Dashboard – interactive data visualisations
# ------------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    # Load combined dataset (red + white) for plotting
    red_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    white_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    red = pd.read_csv(red_url, sep=';')
    white = pd.read_csv(white_url, sep=';')
    red['type'] = 'Red'
    white['type'] = 'White'
    df = pd.concat([red, white], ignore_index=True)
    df['quality_label'] = df['quality'].apply(lambda x: 'Acceptable (>=6)' if x >= 6 else 'Not acceptable (<=5)')

    # 1. Alcohol distribution by quality and type
    fig1 = px.box(df, x='quality_label', y='alcohol', color='type',
                  title='Alcohol Distribution by Quality and Wine Type',
                  color_discrete_map={'Red': '#8B0000', 'White': '#F0E68C'})
    plot1 = plot(fig1, output_type='div', include_plotlyjs='cdn')

    # 2. Correlation heatmap (numeric features only)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig2 = px.imshow(corr, text_auto=True, aspect="auto",
                     title="Feature Correlation Heatmap")
    plot2 = plot(fig2, output_type='div')

    # 3. Feature importance from the trained Random Forest
    rf_model = model.named_steps['clf']
    importances = rf_model.feature_importances_
    # Feature names as used in the model (including 'type')
    feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    fig3 = px.bar(feat_imp_df, x='importance', y='feature', orientation='h',
                  title='Feature Importance (Random Forest)',
                  labels={'importance': 'Importance', 'feature': ''})
    plot3 = plot(fig3, output_type='div')

    # 4. Quality distribution by type
    quality_counts = df.groupby(['type', 'quality_label']).size().reset_index(name='count')
    fig4 = px.bar(quality_counts, x='type', y='count', color='quality_label',
                  title='Quality Distribution by Wine Type',
                  barmode='group',
                  color_discrete_map={'Acceptable (>=6)': 'green', 'Not acceptable (<=5)': 'red'})
    plot4 = plot(fig4, output_type='div')

    return render_template('dashboard.html',
                           plot1=plot1, plot2=plot2, plot3=plot3, plot4=plot4)

if __name__ == '__main__':
    app.run(debug=True)