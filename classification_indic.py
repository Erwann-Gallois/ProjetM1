import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from collections import Counter
from bdd_script import get_indicateur, get_labels
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


# Définir les hyperparamètres à tester pour la recherche sur grille
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 100],  # Valeurs de régularisation
    'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'newton-cholesky'],  # Solvers possibles
    'class_weight': ['balanced', None],  # Poids des classes
    'max_iter': [10, 100, 200, 500, 1000],  # Nombre d'itérations
    'dual': [True, False],
    'fit_intercept': [True, False],
    'intercept_scaling': [0.001, 0.1, 0.5, 1]
}

# Initialiser le modèle de régression logistique
log_model = LogisticRegression()

# Initialiser GridSearchCV
grid_search = GridSearchCV(estimator=log_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Entraînement avec recherche sur grille
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres trouvés
print(f"Best parameters: {grid_search.best_params_}")

# Utiliser le meilleur modèle trouvé
best_model = grid_search.best_estimator_
#best_model = LogisticRegression(C=1,max_iter=200,solver='saga').fit(X_train, y_train)

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
