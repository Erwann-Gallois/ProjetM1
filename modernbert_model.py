from transformers import AutoTokenizer, AutoModel, AutoConfig
from imblearn.over_sampling import SMOTE, ADASYN
import torch
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from bdd_script import get_dialoguefr, get_labels, get_indicateur
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Chargement du modèle et du tokenizer
model_name = "almanach/moderncamembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_tokens = AutoConfig.from_pretrained(model_name).max_position_embeddings

# Encodage des dialogues
def encode_long_dialogue(dialogues, tokenizer, model, max_length, device="cpu"):
    embeddings = []
    dim = model.config.hidden_size
    for i, text in enumerate(dialogues):
        if not isinstance(text, str) or text.strip() == "":
            embeddings.append(np.zeros(dim))
            continue
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
                pooled = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(pooled)
        except Exception as e:
            print(f"Erreur à l'index {i} : {e}")
            embeddings.append(np.zeros(dim))
    return np.vstack(embeddings)

# Chargement et préparation des données
def prepare_dataframe(partof):
    raw = get_indicateur(partof)
    df = pd.json_normalize(raw)
    drop_cols = [col for col in df.columns if col.startswith("has_unit.") or col.startswith("ratio_unit.")]
    df = df.drop(columns=drop_cols, errors='ignore')
    df.fillna(df.mean(), inplace=True)
    return df

X_train_tab = prepare_dataframe(3)
X_test_tab = prepare_dataframe(1)

# Alignement
X_test_tab = X_test_tab.reindex(columns=X_train_tab.columns, fill_value=X_train_tab.mean())

# Standardisation
scaler = StandardScaler()
X_train_tab = scaler.fit_transform(X_train_tab)
X_test_tab = scaler.transform(X_test_tab)

# Encodage texte
encoded_train = encode_long_dialogue(get_dialoguefr(3), tokenizer, model, max_tokens, device)
encoded_test = encode_long_dialogue(get_dialoguefr(1), tokenizer, model, max_tokens, device)

X_train = encoded_train
X_test = encoded_test
# Fusion
# X_train = np.concatenate((encoded_train, X_train_tab), axis=1)
# X_test = np.concatenate((encoded_test, X_test_tab), axis=1)

# Labels
y_train = LabelEncoder().fit_transform(get_labels(3))
y_test = LabelEncoder().fit_transform(get_labels(1))

# Rééchantillonnage
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
X_adasyn, y_adasyn = ADASYN(random_state=42).fit_resample(X_train, y_train)

# Entraînement AdaBoost
def eval_adaboost(X_tr, y_tr, name, balanced=True):

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, target_names=["0", "1"])
    matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    return name, report, matrix, auc

results = [
    eval_adaboost(X_train, y_train, "Gradiant Brut"),
    eval_adaboost(X_smote, y_smote, "Gradiant SMOTE", False),
    eval_adaboost(X_adasyn, y_adasyn, "Gradiant ADASYN", False)
]

# Sauvegarde
with open("resultat_model/resultats_classification_Gradiant_modernbert_text.txt", "w") as f:
    for name, report, matrix, auc in results:
        f.write(f"=== Resultats de la classification {name} ===\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))
        f.write(f"\n\nROC AUC Score: {auc:.4f}\n")
        f.write("=" * 50 + "\n\n")
