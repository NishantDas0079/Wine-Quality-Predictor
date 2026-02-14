# app.py (binary version)
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Load model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'wine_quality_binary.pkl')
model = joblib.load(model_path)

feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # same form, no changes needed

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = df[feature_names]
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]  # [prob_class0, prob_class1]
        class_labels = ['not acceptable', 'acceptable']
        result_label = class_labels[prediction]
        response = {
            'quality': result_label,
            'confidence': round(probability[prediction], 3),
            'probabilities': {
                'not acceptable': round(probability[0], 3),
                'acceptable': round(probability[1], 3)
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/result', methods=['POST'])
def result():
    try:
        features = {name: float(request.form[name]) for name in feature_names}
        df = pd.DataFrame([features])
        df = df[feature_names]
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        class_labels = ['not acceptable', 'acceptable']
        result_label = class_labels[prediction]
        
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

if __name__ == '__main__':
    app.run(debug=True)