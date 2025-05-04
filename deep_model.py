import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from bdd_script import get_indicateur, get_labels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
# === 1. Chargement et préparation des données d'entraînement ===
raw_indicateurs = get_indicateur(3)  # Partie 3 = train
indicateurs_df = pd.json_normalize(raw_indicateurs)

# Supprimer les colonnes has_unit.* et ratio_unit.*
colonnes_a_supprimer = [col for col in indicateurs_df.columns if col.startswith("has_unit.") or col.startswith("ratio_unit.")]
indicateurs_numeriques = indicateurs_df.drop(columns=colonnes_a_supprimer)

# Imputation des NaN par la moyenne
indicateurs_numeriques.fillna(indicateurs_numeriques.mean(), inplace=True)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(indicateurs_numeriques)
y_train = get_labels(3)

# Conversion en tenseurs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

# === 2. Modèle ===
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 3. Entraînement ===
losses = []
for epoch in range(200):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    losses.append(loss.item())

def moving_average(x, window=5):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# === 5. Évaluation sur données de validation ===
raw_val = get_indicateur(2)
val_raw_df = pd.json_normalize(raw_val)

# Supprime les colonnes à ignorer (même si absentes)
val_df = val_raw_df.drop(columns=colonnes_a_supprimer, errors="ignore")

# Reindexe pour avoir exactement les mêmes colonnes que l'entraînement
val_df = val_df.reindex(columns=indicateurs_numeriques.columns)

# Impute les NaN (créés par reindex + colonnes partiellement remplies)
val_df.fillna(val_df.mean(), inplace=True)

# Applique le scaler entraîné
X_val = scaler.transform(val_df)
y_val = get_labels(2)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    logits = model(X_val_tensor)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs >= 0.5).astype(int)

# === 6. Résultats ===
print("\n📊 Classification report sur validation :")
print(classification_report(y_val, preds))

print("\n🎯 Matrice de confusion :")
print(confusion_matrix(y_val, preds))

fpr, tpr, _ = roc_curve(y_val, probs)
auc = roc_auc_score(y_val, probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', lw=0.75)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC")
plt.legend()
plt.grid()
plt.show()