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

import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

def plot_log_loss_curve_with_split(model, X_train, y_train, X_val, y_val, nom_fichier):
    """
    Affiche la courbe de perte log_loss pour un modèle sklearn avec staged_predict_proba.

    Paramètres :
        model : modèle sklearn (AdaBoost, GradientBoosting...) avec staged_predict_proba.
        X_train, y_train : données d'entraînement.
        X_val, y_val : données de validation.
    """
    model.fit(X_train, y_train)

    train_losses = []
    val_losses = []

    if not hasattr(model, "staged_predict_proba"):
        raise ValueError("Le modèle doit avoir une méthode 'staged_predict_proba'.")

    for probas_train, probas_val in zip(model.staged_predict_proba(X_train),
                                        model.staged_predict_proba(X_val)):
        train_losses.append(log_loss(y_train, probas_train))
        val_losses.append(log_loss(y_val, probas_val))

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss", color="orange")
    plt.plot(val_losses, label="Validation Loss", color="orangered")
    plt.xlabel("Nombre d’estimateurs")
    plt.ylabel("Log Loss")
    plt.title("Courbe de perte AdaBoost (log loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"resultat_model/{nom_fichier}.png")



# === Données d'entraînement ===
raw_train = get_indicateur(3)
train_df = pd.json_normalize(raw_train)
colonnes_a_supprimer = [col for col in train_df.columns if col.startswith("has_unit.") or col.startswith("ratio_unit.")]
train_df = train_df.drop(columns=colonnes_a_supprimer)
train_df.fillna(train_df.mean(), inplace=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df)
y_train = get_labels(3)

# === Données de validation ===
raw_test = get_indicateur(1)
val_df = pd.json_normalize(raw_test).drop(columns=colonnes_a_supprimer, errors='ignore')
val_df = val_df.reindex(columns=train_df.columns)
val_df.fillna(val_df.mean(), inplace=True)
X_val = scaler.transform(val_df)
y_val = get_labels(1)


# === Balancement ===
sm = SMOTE()
X_res, y_res = sm.fit_resample(X_train, y_train)

# === ADASYN ===
adasyn = ADASYN()
X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X_train, y_train)


# === AdaBoost Non balancer===
base_estimator = DecisionTreeClassifier(max_depth=1, class_weight=None)
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=10, learning_rate=1.0)
clf_boost.fit(X_train, y_train)

# Prédictions
y_pred = clf_boost.predict(X_val)
probs = clf_boost.predict_proba(X_val)[:, 1]  # pour l'AUC

# Rapport de classification
report = classification_report(y_val, y_pred, target_names=["0", "1"])
conf_matrix = confusion_matrix(y_val, y_pred)
auc_score = roc_auc_score(y_val, probs)

# Courbe de perte
plot_log_loss_curve_with_split(clf_boost, X_train, y_train, X_val, y_val, "courbe_perte_adaboost")

# === AdaBoost balancer ===
base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=10, learning_rate=1.0)
clf_boost.fit(X_res, y_res)

# Prédictions
y_pred_smote = clf_boost.predict(X_val)
probs_smote = clf_boost.predict_proba(X_val)[:, 1]  # pour l'AUC

# Rapport de classification
report_bal = classification_report(y_val, y_pred_smote, target_names=["0", "1"])
conf_matrix_bal = confusion_matrix(y_val, y_pred_smote)
auc_score_bal = roc_auc_score(y_val, probs_smote)

# Courbe de perte
plot_log_loss_curve_with_split(clf_boost, X_res, y_res, X_val, y_val, "courbe_perte_adaboost_smote")

# === ADABoost avec balancement Adasynx ===
base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=10, learning_rate=1.0)
clf_boost.fit(X_res_adasyn, y_res_adasyn)

# Prédictions
y_pred_adasyn = clf_boost.predict(X_val)
probs_adasyn = clf_boost.predict_proba(X_val)[:, 1]  # pour l'AUC
# Rapport de classification
report_adasyn = classification_report(y_val, y_pred_adasyn, target_names=["0", "1"])
conf_matrix_adasyn = confusion_matrix(y_val, y_pred_adasyn)
auc_score_adasyn = roc_auc_score(y_val, probs_adasyn)

# Courbe de perte
plot_log_loss_curve_with_split(clf_boost, X_res_adasyn, y_res_adasyn, X_val, y_val, "courbe_perte_adaboost_adasyn")

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