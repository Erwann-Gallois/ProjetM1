import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import json
import psycopg2


connexion = psycopg2.connect(
    dbname="textes_db",
    user="utilisateur",
    password="mdp",
    host="localhost",
    port="5432"
)
curseur = connexion.cursor()
sql = """SELECT indicateur, phq_binary FROM participants WHERE partof = %s"""
curseur.execute(sql, (1,))

resultats = curseur.fetchall()

if resultats:
    df = pd.DataFrame(resultats, columns=['indicateur', 'phq_binary'])
    df['indicateur'] = df['indicateur'].fillna('{}')
    indicateurs_df = df["indicateur"].apply(pd.Series)
    indicateurs_df["phq_binary"] = df["phq_binary"]
else:
    print("Aucun résultat trouvé.")


X = indicateurs_df.drop(columns=["phq_binary"])
y = indicateurs_df["phq_binary"]

# Convertir toutes les colonnes de X en numérique
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
# n_estimators=100 → Nombre d'arbres dans la forêt
# random_state=42 → Assure la reproductibilité des résultats
# max_depth=None → Laisse les arbres croître complètement
# min_samples_split=2 → Minimum d’échantillons pour diviser un nœud

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle: {accuracy:.2f}")
print(classification_report(y_test, y_pred, zero_division=1))

importances = rf.feature_importances_
features = X.columns
tree = rf.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=["Class 0", "Class 1"], filled=True)
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(features, importances, color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Importance des indicateurs")
plt.show()





#---------------------------------arbre de décision---------------------------

# model = DecisionTreeClassifier(max_depth=3, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# plt.figure(figsize=(12, 6))
# plot_tree(model, feature_names=features, class_names=['Non Dépressif', 'Dépressif'], filled=True)
# plt.show()