import os
import json
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
from tensorflow.keras import backend as K

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

# === SMOTE pour suréchantillonner la classe minoritaire ===
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === Poids de classe dynamiques ===
class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weight))
print("Poids de classe ajustés :", class_weight_dict)

# === Focal Loss ===
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = K.clip(y_true, epsilon, 1. - epsilon)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# === Modèle Keras optimisé ===
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_res.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Sortie binaire pour PHQ_Binary
])

# Optimisation avec Adam et ajustement du learning rate
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', patience=5, factor=0.5)

# === Entraînement ===
history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Visualisation des courbes d'apprentissage ===
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['auc'], label='AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.legend()
plt.title("Courbes d'apprentissage")
plt.xlabel("Epochs")
plt.ylabel("Metrics")
plt.show()

# === Évaluation ===
results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, AUC: {results[2]:.4f}")

# === Prédictions sur le test ===
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# === Matrice de confusion ===
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de Confusion :\n", conf_matrix)

# === Visualisation de la matrice de confusion ===
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()

# === Rapport de classification ===
report = classification_report(y_test, y_pred, target_names=["Classe 0", "Classe 1"])
print("Rapport de Classification :\n", report)

# === Courbe ROC ===
fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.show()

# === AUC ===
print(f"AUC : {roc_auc:.4f}")
