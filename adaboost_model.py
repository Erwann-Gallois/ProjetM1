import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from bdd_script import get_indicateur, get_labels
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron


# === Données d'entraînement ===
raw_train = get_indicateur(3)
train_df = pd.json_normalize(raw_train)
colonnes_a_supprimer = [col for col in train_df.columns if col.startswith("has_unit.") or col.startswith("ratio_unit.")]
train_df = train_df.drop(columns=colonnes_a_supprimer)
train_df.fillna(train_df.mean(), inplace=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df)
y_train = get_labels(3)

# === Données de test ===
raw_test = get_indicateur(1)
test_df = pd.json_normalize(raw_test).drop(columns=colonnes_a_supprimer, errors='ignore')
test_df = test_df.reindex(columns=train_df.columns)
test_df.fillna(test_df.mean(), inplace=True)
X_test = scaler.transform(test_df)
y_test = get_labels(1)


# === Balancement ===
sm = SMOTE()
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"Nombre d'échantillons avant SMOTE: {np.bincount(y_train)}")
print(f"Nombre d'échantillons après SMOTE: {np.bincount(y_res)}")

# === ADASYN ===
adasyn = ADASYN()
X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X_train, y_train)
print(f"Nombre d'échantillons avant ADASYN: {np.bincount(y_train)}")
print(f"Nombre d'échantillons après ADASYN: {np.bincount(y_res_adasyn)}")


# === AdaBoost Non balancer===
base_estimator = DecisionTreeClassifier(max_depth=1)
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=60, learning_rate=1.0)
clf_boost.fit(X_train, y_train)

# Prédictions
y_pred = clf_boost.predict(X_test)
probs = clf_boost.predict_proba(X_test)[:, 1]  # pour l'AUC

# Rapport de classification
report = classification_report(y_test, y_pred, target_names=["0", "1"])
conf_matrix = confusion_matrix(y_test, y_pred)
auc_score = roc_auc_score(y_test, probs)

# === AdaBoost balancer ===
base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=60, learning_rate=1.0)
clf_boost.fit(X_res, y_res)

# Prédictions
y_pred_smote = clf_boost.predict(X_test)
probs_smote = clf_boost.predict_proba(X_test)[:, 1]  # pour l'AUC

# Rapport de classification
report_bal = classification_report(y_test, y_pred_smote, target_names=["0", "1"])
conf_matrix_bal = confusion_matrix(y_test, y_pred_smote)
auc_score_bal = roc_auc_score(y_test, probs_smote)

# === ADABoost avec balancement Adasynx ===
base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=60, learning_rate=1.0)
clf_boost.fit(X_res_adasyn, y_res_adasyn)

# Prédictions
y_pred_adasyn = clf_boost.predict(X_test)
probs_adasyn = clf_boost.predict_proba(X_test)[:, 1]  # pour l'AUC
# Rapport de classification
report_adasyn = classification_report(y_test, y_pred_adasyn, target_names=["0", "1"])
conf_matrix_adasyn = confusion_matrix(y_test, y_pred_adasyn)
auc_score_adasyn = roc_auc_score(y_test, probs_adasyn)

# Écriture dans un fichier texte
with open("resultat_model/resultats_classification_adaboost.txt", "w") as f:
    f.write("=== Résultats de la classification ADABoost non balancée ===\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write(f"\n\nROC AUC Score: {auc_score:.4f}")
    f.write("\n\n" + "="*50 + "\n\n")
    f.write("=== Résultats de la classification ADABoost balancée ===\n")
    f.write("Classification Report:\n")
    f.write(report_bal)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix_bal))
    f.write(f"\n\nROC AUC Score: {auc_score_bal:.4f}")
    f.write("\n\n" + "="*50 + "\n\n")
    f.write("=== Résultats de la classification ADABoost avec ADASYN ===\n")
    f.write("Classification Report:\n")
    f.write(report_adasyn)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix_adasyn))
    f.write(f"\n\nROC AUC Score: {auc_score_adasyn:.4f}")