import pandas as pd
import numpy as np

master_df = pd.read_csv('dataset/processed/master_patient_manifest.csv')
test_clean = pd.read_csv('dataset/processed/test_patient_clean.csv')

features = ['age', 'hba1c_pct', 'systolic_bp', 'diastolic_bp', 'intraocular_pressure_mmhg', 'visual_acuity_logmar', 'diabetes_duration_years']

print('=== EMPIRICAL MEANS IN MASTER PATIENT MANIFEST BY CLASS ===')
summary_master = master_df.groupby('class')[features].agg(['mean', 'std']).round(2)
print(summary_master)

print('\n=== EMPIRICAL SEX DISTRIBUTION BY CLASS ===')
sex_dist = pd.crosstab(master_df['class'], master_df['sex'], normalize='index').round(3)
print(sex_dist)

print('\n=== EMPIRICAL SMOKING STATUS DISTRIBUTION BY CLASS ===')
smoke_dist = pd.crosstab(master_df['class'], master_df['smoking_status'], normalize='index').round(3)
print(smoke_dist)

print('\n=== CHECK CORRELATION OF TABULAR FEATURES WITH GROUND TRUTH CLASS ===')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train a simple random forest on ONLY tabular features to predict disease class
X_tr = pd.get_dummies(master_df[['age', 'sex', 'hba1c_pct', 'systolic_bp', 'diastolic_bp', 'intraocular_pressure_mmhg', 'visual_acuity_logmar', 'diabetes_duration_years', 'smoking_status']], drop_first=True)
y_tr = master_df['class']

X_te = pd.get_dummies(test_clean[['age', 'sex', 'hba1c_pct', 'systolic_bp', 'diastolic_bp', 'intraocular_pressure_mmhg', 'visual_acuity_logmar', 'diabetes_duration_years', 'smoking_status']], drop_first=True)
y_te = test_clean['class']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)
preds = rf.predict(X_te)
print('\nClassification Report using ONLY Synthetic Tabular Features on Clean Test Split:')
print(classification_report(y_te, preds, digits=4))
