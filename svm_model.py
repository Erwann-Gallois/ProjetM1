import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from bdd_script import get_indicateur, get_labels
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# === Chargement des données ===
raw_train = get_indicateur(3)
raw_test = get_indicateur(1)

train_df = pd.json_normalize(raw_train)
test_df = pd.json_normalize(raw_test)

colonnes_a_supprimer = [col for col in train_df.columns if col.startswith("has_unit.") or col.startswith("ratio_unit.")]

train_df = train_df.drop(columns=colonnes_a_supprimer, errors='ignore') # Utilisez errors='ignore' pour éviter les erreurs si les colonnes n'existent pas
train_df.fillna(train_df.mean(), inplace=True)

test_df = test_df.drop(columns=colonnes_a_supprimer, errors='ignore') # Utilisez errors='ignore' pour éviter les erreurs si les colonnes n'existent pas
test_df = test_df.reindex(columns=train_df.columns, fill_value= train_df.mean()) # Utilisez fill_testue pour aligner les colonnes et gérer les testeurs manquantes

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df)
X_test = scaler.transform(test_df)
print(f"Shape des données d'entraînement : {X_train.shape}")
print(f"Shape des données de test : {X_test.shape}")

# Encodage des labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(get_labels(3))
y_test = encoder.transform(get_labels(1))

# === Normalisation ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Balancement ===
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# === ADASYN ===
adasyn = ADASYN(random_state=42)
X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X_train, y_train)


# === AdaBoost Non balancer===
svm_clf = SVC(C = 7, gamma = 0.3, kernel= 'sigmoid', probability= True, random_state=42, class_weight=None)
svm_clf.fit(X_train, y_train)

# Prédictions
y_pred = svm_clf.predict(X_test)
probs = svm_clf.predict_proba(X_test)[:, 1]  # pour l'AUC

# Rapport de classification
report = classification_report(y_test, y_pred, target_names=["0", "1"])
conf_matrix = confusion_matrix(y_test, y_pred)
auc_score = roc_auc_score(y_test, probs)

# Courbe de perte
# plot_log_loss_curve_with_split(svm_clf, X_train, y_train, X_val, y_val, "courbe_perte_adaboost")

# === AdaBoost balancer ===
svm_clf = SVC(C = 7, gamma = 0.3, kernel= 'sigmoid', probability= True, random_state=42,  class_weight=None)
svm_clf.fit(X_res, y_res)

# Prédictions
y_pred_smote = svm_clf.predict(X_test)
probs_smote = svm_clf.predict_proba(X_test)[:, 1]  # pour l'AUC

# Rapport de classification
report_bal = classification_report(y_test, y_pred_smote, target_names=["0", "1"])
conf_matrix_bal = confusion_matrix(y_test, y_pred_smote)
auc_score_bal = roc_auc_score(y_test, probs_smote)

# Courbe de perte
# plot_log_loss_curve_with_split(svm_clf, X_res, y_res, X_val, y_val, "courbe_perte_adaboost_smote")

# === ADABoost avec balancement Adasynx ===
svm_clf = SVC(C = 7, gamma = 0.3, kernel= 'sigmoid', probability= True, random_state=42,  class_weight=None)
svm_clf.fit(X_res_adasyn, y_res_adasyn)

# Prédictions
y_pred_adasyn = svm_clf.predict(X_test)
probs_adasyn = svm_clf.predict_proba(X_test)[:, 1]  # pour l'AUC
# Rapport de classification
report_adasyn = classification_report(y_test, y_pred_adasyn, target_names=["0", "1"])
conf_matrix_adasyn = confusion_matrix(y_test, y_pred_adasyn)
auc_score_adasyn = roc_auc_score(y_test, probs_adasyn)
import joblib
joblib.dump(svm_clf, 'svm.pkl')

# Courbe de perte
# plot_log_loss_curve_with_split(svm_clf, X_res_adasyn, y_res_adasyn, X_val, y_val, "courbe_perte_adaboost_adasyn")

# Écriture dans un fichier texte
with open("resultat_model/resultats_classification_SVM.txt", "w") as f:
    f.write("=== Résultats de la classification SVM non balancée ===\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write(f"\n\nROC AUC Score: {auc_score:.4f}")
    f.write("\n\n" + "="*50 + "\n\n")
    f.write("=== Résultats de la classification SVM balancée ===\n")
    f.write("Classification Report:\n")
    f.write(report_bal)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix_bal))
    f.write(f"\n\nROC AUC Score: {auc_score_bal:.4f}")
    f.write("\n\n" + "="*50 + "\n\n")
    f.write("=== Résultats de la classification SVM avec ADASYN ===\n")
    f.write("Classification Report:\n")
    f.write(report_adasyn)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix_adasyn))
    f.write(f"\n\nROC AUC Score: {auc_score_adasyn:.4f}")