import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder

# Chargement des données (remplace cette ligne par ton propre jeu de données)
# Exemple : data = pd.read_csv('ton_fichier.csv')
# Assure-toi que la colonne 'target' correspond à ta variable cible
# data = pd.read_csv('path_to_your_data.csv')

# Séparation des features et de la cible
# === Chargement des données JSON ===
DATA_DIR = "data_model"

def load_json_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            first_char = file.read(1)
            file.seek(0)
            return json.load(file) if first_char == "[" else [json.loads(line) for line in file]
    except Exception as e:
        print(f"Erreur lors du chargement de {filename}: {e}")
        return []

train_data = load_json_data("train.json")
test_data = load_json_data("test.json")
val_data = load_json_data("validation.json")

if not train_data or not test_data or not val_data:
    raise ValueError("Les données n'ont pas été chargées correctement.")

# === Préparation des données ===
def prepare_data(data):
    texts = [" ".join(entry["dialogue"]) for entry in data]
    labels = [entry["PHQ_Binary"] for entry in data]
    return texts, labels

train_texts, train_labels = prepare_data(train_data)
test_texts, test_labels = prepare_data(test_data)
val_texts, val_labels = prepare_data(val_data)

# === Encodage des labels ===
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_labels)
y_test = encoder.transform(test_labels)
y_val = encoder.transform(val_labels)

def afficher_distribution(y, nom_set):
    classes, counts = np.unique(y, return_counts=True)
    print(f"Distribution dans {nom_set} :")
    for c, n in zip(classes, counts):
        print(f"  Classe {c} : {n} exemples")
    print()

afficher_distribution(y_train, "Train")
afficher_distribution(y_test, "Test")
afficher_distribution(y_val, "Validation")

# === TF-IDF Vectorisation améliorée ===
vectorizer = TfidfVectorizer(max_features=8000, stop_words='english', ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()
X_val = vectorizer.transform(val_texts).toarray()

# Calcul des poids de classe (important pour les classes déséquilibrées)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Modèle XGBoost
xgb_model = XGBClassifier(scale_pos_weight=sample_weights.sum() / (y_train == 1).sum(), use_label_encoder=False, eval_metric='logloss')

# Modèle Random Forest
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)

# Modèle d'ensemble (Voting Classifier)
ensemble_model = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model)
], voting='soft')

# Entraînement du modèle d'ensemble
ensemble_model.fit(X_train, y_train, sample_weight=sample_weights)

# Prédictions sur le jeu de test
y_pred = ensemble_model.predict(X_test)
y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]

# Évaluation du modèle
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Optionnel: Visualisation de la matrice de confusion
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Prédictions')
plt.ylabel('Véritables Labels')
plt.title('Matrice de Confusion')
plt.show()
