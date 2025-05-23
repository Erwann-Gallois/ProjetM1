import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
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

curseur.execute(sql, (3,))
train = curseur.fetchall()

curseur.execute(sql, (1,))
test = curseur.fetchall()

# data = np.concatenate([train, test])

# df = pd.DataFrame(data, columns=['indicateur', 'phq_binary'])
# df['indicateur'] = df['indicateur'].fillna('{}')
# indicateurs_df = df["indicateur"].apply(pd.Series)
# indicateurs_df["phq_binary"] = df["phq_binary"]

# X = indicateurs_df.drop(columns=["phq_binary"])
# y = indicateurs_df["phq_binary"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
y_train = y_train.apply(pd.to_numeric, errors="coerce").fillna(0)
y_test = y_test.apply(pd.to_numeric, errors="coerce").fillna(0)

print(X_test.shape)
print(X_train.shape)

rf1 = RandomForestClassifier(n_estimators=100, random_state=42)

rf1.fit(X_train, y_train)

y_pred1 = rf1.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred1)
print(f"Précision du modèle non balance: {accuracy1:.2f}")
print(classification_report(y_test, y_pred1, zero_division=1))
print(confusion_matrix(y_test, y_pred1))

sm = SMOTE()
X_res, y_res = sm.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)#, class_weight='balanced')

rf.fit(X_res, y_res)

# rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle balance: {accuracy:.2f}")
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))

importances = rf.feature_importances_
features = X_train.columns
tree = rf.estimators_[0]

# plt.figure(figsize=(20, 10))
# plot_tree(tree, feature_names=X_train.columns, class_names=["Class 0", "Class 1"], filled=True)
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.barh(features, importances, color="skyblue")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.title("Importance des indicateurs")
# plt.show()



feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
important_features = feature_importances[feature_importances >= 0.02].index

X_train_filtered = X_train[important_features]
X_test_filtered = X_test[important_features]

rf_filtered = RandomForestClassifier(random_state=42)
rf_filtered.fit(X_train_filtered, y_train)

y_pred_filtered = rf_filtered.predict(X_test_filtered)

accuracy_filtered = accuracy_score(y_test, y_pred_filtered)
print(f"Précision du modèle filtre: {accuracy_filtered:.2f}")
print(classification_report(y_test, y_pred_filtered, zero_division=1))
print(confusion_matrix(y_test, y_pred_filtered))

importances_filtered = rf_filtered.feature_importances_
features_filtered = X_train_filtered.columns
tree_filtered = rf_filtered.estimators_[0]

# plt.figure(figsize=(20, 10))
# plot_tree(tree_filtered, feature_names=X_train_filtered.columns, class_names=["Class 0", "Class 1"], filled=True)
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.barh(features_filtered, importances_filtered, color="skyblue")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.title("Importance des indicateurs filtre")
# plt.show()

curseur.close()