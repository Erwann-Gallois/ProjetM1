import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from bdd_script import get_indicateur, get_labels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# === Chargement des données ===
raw_train = get_indicateur(3)
train_df = pd.json_normalize(raw_train)

# Suppression des colonnes inutiles
colonnes_a_supprimer = [col for col in train_df.columns if col.startswith("has_unit.") or col.startswith("ratio_unit.")]
train_df = train_df.drop(columns=colonnes_a_supprimer, errors='ignore')

# On remplit les NaN par la moyenne
train_df.fillna(train_df.mean(), inplace=True)

# Chargement des données de test
raw_test = get_indicateur(1)
val_df = pd.json_normalize(raw_test).drop(columns=colonnes_a_supprimer, errors='ignore')

# Assurez-vous que les étiquettes sont bien des entiers et sans NaN
y_train = get_labels(3)
y_val = get_labels(1)

# Vérification de la présence de NaN et conversion en entiers si nécessaire
y_train = pd.Series(y_train).fillna(0).astype(int).values
y_val = pd.Series(y_val).fillna(0).astype(int).values

# === Division stratifiée des données ===
X_train, X_val, y_train, y_val = train_test_split(
    train_df, y_train, test_size=0.3, stratify=y_train, random_state=42
)

# === Normalisation ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# === Calcul des poids de classe ===
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Modèle XGBoost
xgb_model = XGBClassifier(scale_pos_weight=sample_weights.sum() / (y_train == 1).sum(), use_label_encoder=False, eval_metric='logloss')
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# === Voting Classifier ===
ensemble_model = VotingClassifier(estimators=[('xgb', xgb_model), ('rf', rf_model)], voting='soft')

# === Grille de recherche ===
param_grid = {
    # XGBoost Parameters
    'xgb__n_estimators': [5],
    'xgb__max_depth': [3],
    'xgb__learning_rate': [0.3],
    'xgb__subsample': [1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    'xgb__gamma': [0, 0.1, 0.2, 0.5],
    'xgb__min_child_weight': [1, 5, 10, 20],
    'xgb__scale_pos_weight' : [1, 2, 3, 5],
    

    # Random Forest Parameters
    'rf__n_estimators': [100],
    'rf__max_depth': [1],
    'rf__min_samples_split': [2],
    'rf__min_samples_leaf': [1],
    'rf__max_features': ['auto', 'sqrt', 'log2'],
    'rf__bootstrap': [True, False],
    'rf__class_weight' : ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]

}

# === Grid Search ===
grid_search = GridSearchCV(
    estimator=ensemble_model,
    param_grid=param_grid,
    scoring='f1_macro',  # Utilisation de 'f1' pour privilégier un bon compromis entre recall et precision
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Entraînement avec échantillons pondérés
grid_search.fit(X_train, y_train, **{'sample_weight': sample_weights})

# Résultats
best_model = grid_search.best_estimator_
print("Meilleurs paramètres :", grid_search.best_params_)

# Prédictions
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# Évaluation
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("AUC Score:", roc_auc_score(y_val, y_pred_proba))

# Matrice de confusion
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Prédictions')
plt.ylabel('Véritables Labels')
plt.title('Matrice de Confusion')
plt.show()

# Classification Report:

import joblib
joblib.dump(best_model, 'voting.pkl')