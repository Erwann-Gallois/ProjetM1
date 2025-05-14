import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from bdd_script import get_indicateur, get_labels
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Chargement des données
raw_train = get_indicateur(3)
train_df = pd.json_normalize(raw_train)
colonnes_a_supprimer = [col for col in train_df.columns if col.startswith("has_unit.") or col.startswith("ratio_unit.")]
train_df = train_df.drop(columns=colonnes_a_supprimer, errors='ignore') # Utilisez errors='ignore' pour éviter les erreurs si les colonnes n'existent pas
train_df.fillna(train_df.mean(), inplace=True)

raw_test = get_indicateur(1)
test_df = pd.json_normalize(raw_test).drop(columns=colonnes_a_supprimer, errors='ignore')
test_df = test_df.reindex(columns=train_df.columns, fill_value=train_df.mean()) # Utilisez fill_value pour aligner les colonnes et gérer les valeurs manquantes
X_train = train_df
y_train = get_labels(3)
X_test = test_df
y_test = get_labels(1)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Balancement
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle non balancer: {accuracy:.2f}")
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(model, 'grad_boost.pkl')

# balancer
model_bal = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_res)

# Train model
model_bal.fit(X_res, y_res, sample_weight=sample_weights)

# Predict
y_pred_bal = model_bal.predict(X_test)

accuracy_bal = accuracy_score(y_test, y_pred_bal)
print(f"Précision du modèle balancer: {accuracy_bal:.2f}")
print(classification_report(y_test, y_pred_bal, zero_division=1))
print(confusion_matrix(y_test, y_pred_bal))