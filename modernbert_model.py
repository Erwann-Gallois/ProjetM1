from transformers import AutoTokenizer, AutoModel, AutoConfig
from imblearn.over_sampling import SMOTE
import torch
import numpy as np
from bdd_script import get_dialoguefr, get_labels  # Ton script personnalisé
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score


# Chargement du modèle et du tokenizer
model_name = "almanach/moderncamembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Longueur maximale autorisée
max_tokens = config.max_position_embeddings

# Chargement des dialogues
dialogues_train = get_dialoguefr(3)
dialogues_val = get_dialoguefr(2)

# Fonction d'encodage
def encode_long_dialogue(dialogues, tokenizer, model, max_length=8192, device="cpu"):
    embeddings = []
    dim = model.config.hidden_size  # généralement 768

    for i, text in enumerate(dialogues):
        if not isinstance(text, str) or text.strip() == "":
            # Dialogue vide : vecteur nul
            embeddings.append(np.zeros(dim))
            continue

        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            if mean_embedding.shape != (dim,):
                print(f"⚠️ Embedding inattendu à l'index {i}: shape = {mean_embedding.shape}")
                embeddings.append(np.zeros(dim))
            else:
                embeddings.append(mean_embedding)

        except Exception as e:
            print(f"❌ Erreur à l'encodage du dialogue {i}: {e}")
            embeddings.append(np.zeros(dim))

    return np.vstack(embeddings)  # assure un array 2D de shape (N, dim)


# Exécution
encoded_dialogues_train = encode_long_dialogue(dialogues_train, tokenizer, model, max_length=max_tokens, device=device)
encoded_dialogues_val = encode_long_dialogue(dialogues_val, tokenizer, model, max_length=max_tokens, device=device)

print(f"Shape des embeddings d'entraînement : {encoded_dialogues_train.shape}")
print(f"Shape des embeddings de validation : {encoded_dialogues_val.shape}")

# scaler = StandardScaler()
# X_train = scaler.fit_transform(encoded_dialogues_train)
# X_val = scaler.transform(encoded_dialogues_val)
X_train = encoded_dialogues_train
X_val = encoded_dialogues_val
y_train = get_labels(3)
print(f"Shape des labels d'entraînement : {y_train.shape}")
y_val = get_labels(2)
print(f"Shape des labels de validation : {y_val.shape}")

sm = SMOTE()
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
print(f"Shape des données équilibrées : {X_train_balanced.shape}, {y_train_balanced.shape}")

# === AdaBoost Non balancer===
base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=200, learning_rate=0.01)
clf_boost.fit(X_train, y_train)

print("=== AdaBoost Non balancer ===")
print("Rapport de classification :")
print(classification_report(y_val, clf_boost.predict(X_val), target_names=["0", "1"]))
print("Matrice de confusion :")
print(confusion_matrix(y_val, clf_boost.predict(X_val)))
print("AUC :")
print(roc_auc_score(y_val, clf_boost.predict_proba(X_val)[:, 1]))
print("F1 Score :")
print(f1_score(y_val, clf_boost.predict(X_val), average='macro'))

# === AdaBoost balancer===
base_estimator = DecisionTreeClassifier(random_state=42, class_weight=None)
clf_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=200, learning_rate=0.01)
clf_boost.fit(X_train_balanced, y_train_balanced)

print("=== AdaBoost balancer ===")
print("Rapport de classification :")
print(classification_report(y_val, clf_boost.predict(X_val), target_names=["0", "1"]))
print("Matrice de confusion :")
print(confusion_matrix(y_val, clf_boost.predict(X_val)))
print("AUC :")
print(roc_auc_score(y_val, clf_boost.predict_proba(X_val)[:, 1]))
print("F1 Score :")
print(f1_score(y_val, clf_boost.predict(X_val), average='macro'))

