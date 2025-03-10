import os
import json
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Définir le chemin vers les fichiers
DATA_DIR = "data_model"

# Fonction pour charger les fichiers JSON (gère JSON et JSONL)
def load_json_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            first_char = file.read(1)
            file.seek(0)  # Retourner au début du fichier

            if first_char == "[":  # JSON classique (liste d'objets)
                return json.load(file)
            else:  # JSONL (une ligne = un objet JSON)
                return [json.loads(line) for line in file]
    except Exception as e:
        print(f"Erreur lors du chargement de {filename}: {e}")
        return []

# Charger les données
train_data = load_json_data("train.json")
test_data = load_json_data("test.json")
val_data = load_json_data("validation.json")

# Vérification du chargement
if not train_data or not test_data or not val_data:
    raise ValueError("Les données n'ont pas été chargées correctement. Vérifiez les fichiers JSON.")

# Fonction pour préparer les données (textes et labels)
def prepare_data(data):
    texts = [" ".join(entry["dialogue"]) for entry in data]  # Fusionne les phrases du dialogue
    labels = [entry["PHQ_Binary"] for entry in data]  # Cible (0 ou 1)
    return texts, labels

# Préparer les données pour chaque dataset
train_texts, train_labels = prepare_data(train_data)
test_texts, test_labels = prepare_data(test_data)
val_texts, val_labels = prepare_data(val_data)

# Encodage des labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_labels)
y_test = encoder.transform(test_labels)
y_val = encoder.transform(val_labels)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
X_val = vectorizer.transform(val_texts)

# Définir les hyperparamètres à tester pour la recherche sur grille
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Valeurs de régularisation
    'solver': ['liblinear', 'saga'],  # Solvers possibles
    'class_weight': ['balanced', None],  # Poids des classes
    'max_iter': [200, 500, 1000]  # Nombre d'itérations
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

# Évaluation sur les données de test
y_pred_log = best_model.predict(X_test)
print("Logistic Regression - Test Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report for Test Data:")
print(classification_report(y_test, y_pred_log))

# Évaluation sur les données de validation
y_pred_val = best_model.predict(X_val)
print("Classification Report for Validation Data:")
print(classification_report(y_val, y_pred_val))

# Validation croisée avec le meilleur modèle
cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Logistic Regression - Validation Accuracy (cross-val): {np.mean(cross_val_scores):.4f}")

# bon résultat de la part du modèle actuellement pour les cas non depressif, mais difficulté sur les cas depressif qui sont ceux qui nous intéressent
"""Best parameters: {'C': 1, 'class_weight': 'balanced', 'max_iter': 200, 'solver': 'saga'}
Logistic Regression - Test Accuracy: 0.7142857142857143
Classification Report for Test Data:
              precision    recall  f1-score   support

           0       0.79      0.79      0.79        39
           1       0.53      0.53      0.53        17

    accuracy                           0.71        56
   macro avg       0.66      0.66      0.66        56
weighted avg       0.71      0.71      0.71        56

Classification Report for Validation Data:
              precision    recall  f1-score   support

           0       0.85      0.93      0.89        44
           1       0.62      0.42      0.50        12

    accuracy                           0.82        56
   macro avg       0.74      0.67      0.70        56
weighted avg       0.81      0.82      0.81        56

C:\Users\aurel\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\sklearn\linear_model\_sag.py:349: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
Logistic Regression - Validation Accuracy (cross-val): 0.7854"""