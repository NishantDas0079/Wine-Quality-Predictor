# train_combined_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# 1. Load both datasets
red_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
white_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

red = pd.read_csv(red_url, sep=';')
white = pd.read_csv(white_url, sep=';')

# 2. Add type column (0 = red, 1 = white)
red['type'] = 0
white['type'] = 1

# 3. Concatenate
df = pd.concat([red, white], ignore_index=True)

# 4. Create binary target: acceptable (>=6) vs not acceptable (<=5)
df['quality_bin'] = (df['quality'] >= 6).astype(int)  # 1 = acceptable, 0 = not acceptable

# Check distribution
print("Binary class distribution:\n", df['quality_bin'].value_counts())
print("\nWine type distribution:\n", df['type'].value_counts())

# 5. Features (original 11 + 'type')
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates',
    'alcohol', 'type'
]
X = df[feature_names]
y = df['quality_bin']

# 6. Train/test split (stratify to keep class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Preprocessing: scale all features (including 'type', which is binary but scaling doesn't harm)
preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), feature_names)
])

# 8. Pipeline with SMOTE and Random Forest (no class_weight needed because SMOTE balances classes)
pipeline = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

# 9. Train
pipeline.fit(X_train, y_train)

# 10. Evaluate
print("Training accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy:", pipeline.score(X_test, y_test))
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['not acceptable', 'acceptable']))

# 11. Save the model
joblib.dump(pipeline, 'wine_quality_combined.pkl')
print("✅ Combined model saved as wine_quality_combined.pkl")