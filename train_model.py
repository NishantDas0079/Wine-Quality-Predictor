# train_model_binary.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')

# 2. Create binary target: acceptable (>=6) vs not acceptable (<=5)
df['quality_bin'] = (df['quality'] >= 6).astype(int)  # 1 = acceptable, 0 = not acceptable

# Check distribution
print("Binary class distribution:\n", df['quality_bin'].value_counts())

# 3. Features
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]
X = df[feature_names]
y = df['quality_bin']

# 4. Train/test split (stratify to keep class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Pipeline with scaling and classifier (class_weight='balanced' helps)
preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), feature_names)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# 6. Train
pipeline.fit(X_train, y_train)

# 7. Evaluate
print("Training accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy:", pipeline.score(X_test, y_test))
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['not acceptable', 'acceptable']))

# 8. Save the model
joblib.dump(pipeline, 'wine_quality_binary.pkl')
print("✅ Model saved as wine_quality_binary.pkl")