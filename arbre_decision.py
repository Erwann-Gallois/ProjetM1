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
train = curseur.fetchall()

curseur.execute(sql, (3,))
test = curseur.fetchall()

if train and test:
    df_train = pd.DataFrame(train, columns=['indicateur', 'phq_binary'])
    df_train['indicateur'] = df_train['indicateur'].fillna('{}')
    indicateurs_df_train = df_train["indicateur"].apply(pd.Series)
    indicateurs_df_train["phq_binary"] = df_train["phq_binary"]

    df_test = pd.DataFrame(test, columns=['indicateur', 'phq_binary'])
    df_test['indicateur'] = df_test['indicateur'].fillna('{}')
    indicateurs_df_test = df_test["indicateur"].apply(pd.Series)
    indicateurs_df_test["phq_binary"] = df_test["phq_binary"]
else:
    print("Aucun résultat trouvé")


X_train = indicateurs_df_train.drop(columns=["phq_binary"])
y_train = indicateurs_df_train["phq_binary"]

X_test = indicateurs_df_test.drop(columns=["phq_binary"])
y_test = indicateurs_df_test["phq_binary"]

# Convertir toutes les colonnes de X en numérique
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
features = X_train.columns
tree = rf.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X_train.columns, class_names=["Class 0", "Class 1"], filled=True)
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(features, importances, color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Importance des indicateurs")
plt.show()

curseur.close()



#---------------------------------arbre de décision---------------------------

# model = DecisionTreeClassifier(max_depth=3, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# plt.figure(figsize=(12, 6))
# plot_tree(model, feature_names=features, class_names=['Non Dépressif', 'Dépressif'], filled=True)
# plt.show()