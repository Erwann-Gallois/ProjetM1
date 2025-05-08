import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from bdd_script import get_indicateur, get_labels
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# === Chargement des données ===
raw_train = get_indicateur(3)
train_df = pd.json_normalize(raw_train)
colonnes_a_supprimer = [col for col in train_df.columns if col.startswith("has_unit.") or col.startswith("ratio_unit.")]
train_df = train_df.drop(columns=colonnes_a_supprimer, errors='ignore') # Utilisez errors='ignore' pour éviter les erreurs si les colonnes n'existent pas
train_df.fillna(train_df.mean(), inplace=True)

raw_test = get_indicateur(1)
val_df = pd.json_normalize(raw_test).drop(columns=colonnes_a_supprimer, errors='ignore')
val_df = val_df.reindex(columns=train_df.columns, fill_value=train_df.mean()) # Utilisez fill_value pour aligner les colonnes et gérer les valeurs manquantes
X_train = train_df
y_train = get_labels(3)
X_val = val_df
y_val = get_labels(1)

# === Normalisation ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# === Balancement ===
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# === ADASYN ===
adasyn = ADASYN(random_state=42)
X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X_train, y_train)


# === AdaBoost Non balancer===
base_estimator = DecisionTreeClassifier(random_state=42)
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=0.01)
clf_boost.fit(X_train, y_train)

# Prédictions
y_pred = clf_boost.predict(X_val)
probs = clf_boost.predict_proba(X_val)[:, 1]  # pour l'AUC

# Rapport de classification
report = classification_report(y_val, y_pred, target_names=["0", "1"])
conf_matrix = confusion_matrix(y_val, y_pred)
auc_score = roc_auc_score(y_val, probs)
f1_score_non = f1_score(y_val, y_pred, average='weighted')


# === AdaBoost balancer ===
base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=0.01, random_state=42)
clf_boost.fit(X_res, y_res)

# Prédictions
y_pred_smote = clf_boost.predict(X_val)
probs_smote = clf_boost.predict_proba(X_val)[:, 1]  # pour l'AUC

# Rapport de classification
report_bal = classification_report(y_val, y_pred_smote, target_names=["0", "1"])
conf_matrix_bal = confusion_matrix(y_val, y_pred_smote)
auc_score_bal = roc_auc_score(y_val, probs_smote)
f1_score_bal = f1_score(y_val, y_pred_smote, average='weighted')


# === ADABoost avec balancement Adasynx ===
base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=0.01, random_state=42)
clf_boost.fit(X_res_adasyn, y_res_adasyn)

# Prédictions
y_pred_adasyn = clf_boost.predict(X_val)
probs_adasyn = clf_boost.predict_proba(X_val)[:, 1]  # pour l'AUC
# Rapport de classification
report_adasyn = classification_report(y_val, y_pred_adasyn, target_names=["0", "1"])
conf_matrix_adasyn = confusion_matrix(y_val, y_pred_adasyn)
auc_score_adasyn = roc_auc_score(y_val, probs_adasyn)
f1_score_adasyn = f1_score(y_val, y_pred_adasyn, average='weighted')


# Écriture dans un fichier texte
with open("resultat_model/resultats_classification_adaboost.txt", "w") as f:
    f.write("=== Résultats de la classification ADABoost ===\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write(f"\n\nROC AUC Score: {auc_score:.4f}")
    f.write(f"\n\nF1 Score: {f1_score_non:.4f}")
    f.write("\n\n" + "="*50 + "\n\n")
    f.write("=== Résultats de la classification ADABoost SMOTE ===\n")
    f.write("Classification Report:\n")
    f.write(report_bal)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix_bal))
    f.write(f"\n\nROC AUC Score: {auc_score_bal:.4f}")
    f.write(f"\n\nF1 Score: {f1_score_bal:.4f}")
    f.write("\n\n" + "="*50 + "\n\n")
    f.write("=== Résultats de la classification ADABoost avec ADASYN ===\n")
    f.write("Classification Report:\n")
    f.write(report_adasyn)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix_adasyn))
    f.write(f"\n\nROC AUC Score: {auc_score_adasyn:.4f}")
    f.write(f"\n\nF1 Score: {f1_score_adasyn:.4f}")