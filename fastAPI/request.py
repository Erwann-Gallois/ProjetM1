import joblib
import requests
import time
import sys
import os
import spacy
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import morpho
import generate_PDF

url = "http://127.0.0.1:8000/indic/sep"
urlInit = "http://127.0.0.1:8000/init"

# Chargement depuis le fichier
adaboost = joblib.load('models/adaboost.pkl')
grad_boost = joblib.load('models/grad_boost.pkl')
randomForest = joblib.load('models/randomForest.pkl')
regression = joblib.load('models/regression.pkl')
svm = joblib.load('models/svm.pkl')
voting = joblib.load('models/voting.pkl')

# Utilisation comme d’habitude
# predictions = modele_charge.predict(X_test)


csv_file_path = Path(__file__).resolve().parents[1] / "datasets/data_fr/DAMT_FR/FR_D0420-S1-T05.csv"
print(str(csv_file_path))
result_path = os.path.join(os.path.dirname(__file__),"..", "resultat")

if not csv_file_path.exists():
    print(f"Fichier CSV introuvable : {csv_file_path}")
    sys.exit(1)

nlp = spacy.load('fr_core_news_lg')

def segmenter_phrases(texte):
    doc = nlp(texte)
    return [phrase.text.strip() for phrase in doc.sents]

dialogue_brut = morpho.export_patient_dialogue(str(csv_file_path))

if dialogue_brut is None:
    print("Erreur : Dialogue brut introuvable ou vide.")
    sys.exit(1)
    
phrases = segmenter_phrases(dialogue_brut)

if not phrases:
    print("Erreur : Aucune phrase trouvée après segmentation.")
    sys.exit(1)

indicateurs = {
    "TTR": [],
    "taux_adverbes": [],
    "taux_verbes_conjugué": [],
    "densite_concept_total": [],
    "phrases": []
}

emotions = {
    "anger_rate": [],
    "disgust_rate": [],
    "fear_rate": [],
    "joy_rate": [],
    "sadness_rate": [],
    "surprise_rate": [],
    "phrases": []
}

indicateurs_T = {
    "TTR": [],
    "taux_adverbes": [],
    "taux_verbes_conjugué": [],
    "densite_concept_total": [],
    "phrases": []
}

emotions_T = {
    "anger_rate": [],
    "disgust_rate": [],
    "fear_rate": [],
    "joy_rate": [],
    "sadness_rate": [],
    "surprise_rate": [],
    "phrases": []
}

fig, ax = plt.subplots(2, 2)
ax[0, 0].set_xlim(0, 100)
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_xlabel("Numéro de phrase")
ax[0, 0].set_ylabel("Indicateurs cumulé")
line1, = ax[0, 0].plot([], [], label="TTR", color="blue")
line2, = ax[0, 0].plot([], [], label="Taux d'adverbes", color="green")
line3, = ax[0, 0].plot([], [], label="Taux de verbes conjugué", color="red")
line4, = ax[0, 0].plot([], [], label="densite total des concepts", color="purple")
ax[0, 0].legend()
ax[0, 0].set_title("Graph indicateur cumulé")

ax[0, 1].set_xlim(0, 100)
ax[0, 1].set_ylim(0, 0.1)
ax[0, 1].set_xlabel("Numéro de phrase")
ax[0, 1].set_ylabel("Emotions_rate cumulé")
lineE1, = ax[0, 1].plot([], [], label="sadness", color="blue")
lineE2, = ax[0, 1].plot([], [], label="disgust", color="green")
lineE3, = ax[0, 1].plot([], [], label="anger", color="red")
lineE4, = ax[0, 1].plot([], [], label="fear", color="purple")
lineE5, = ax[0, 1].plot([], [], label="joy", color="yellow")
lineE6, = ax[0, 1].plot([], [], label="surprise", color="orange")
ax[0, 1].legend()
ax[0, 1].set_title("Graph emotions cumulé")

ax[1, 0].set_xlim(0, 100)
ax[1, 0].set_ylim(0, 1)
ax[1, 0].set_xlabel("Numéro de phrase")
ax[1, 0].set_ylabel("Indicateurs par phrase")
lineT1, = ax[1, 0].plot([], [], label="TTR", color="blue")
lineT2, = ax[1, 0].plot([], [], label="Taux d'adverbes", color="green")
lineT3, = ax[1, 0].plot([], [], label="Taux de verbes conjugué", color="red")
lineT4, = ax[1, 0].plot([], [], label="densite total des concepts", color="purple")
ax[1, 0].legend()
ax[1, 0].set_title("Phrase seul")

ax[1, 1].set_xlim(0, 100)
ax[1, 1].set_ylim(0, 0.1)
ax[1, 1].set_xlabel("Numéro de phrase")
ax[1, 1].set_ylabel("Emotions_rate par phrase")
lineTE1, = ax[1, 1].plot([], [], label="sadness", color="blue")
lineTE2, = ax[1, 1].plot([], [], label="disgust", color="green")
lineTE3, = ax[1, 1].plot([], [], label="anger", color="red")
lineTE4, = ax[1, 1].plot([], [], label="fear", color="purple")
lineTE5, = ax[1, 1].plot([], [], label="joy", color="yellow")
lineTE6, = ax[1, 1].plot([], [], label="surprise", color="orange")
ax[1, 1].legend()
ax[1, 1].set_title("Phrase seul")

response_init = requests.post(urlInit)
if response_init.status_code == 200:
    print("Temps initialisé avec succès :", response_init.json())
else:
    print("Erreur lors de l'initialisation du temps :", response_init.status_code, response_init.text)
    sys.exit(1)

requete_en_cours = False

last_indic = {
    "TTR": 0,
    "taux_adverbes": 0,
    "taux_verbes_conjugué": 0,
    "densite_concept_total": 0
}

last_emotions = {
    "anger_rate": 0,
    "disgust_rate": 0,
    "fear_rate": 0,
    "joy_rate": 0,
    "sadness_rate": 0,
    "surprise_rate": 0
}

indic = None
def update(frame):
    global requete_en_cours
    
    if frame < len(phrases) and not requete_en_cours:
        phrase = phrases[frame]
        data = {
            "texte": phrase,
            "phrase": [phrase],
            "indicateurs": []
        }

        requete_en_cours = True

        response = requests.post(url, json=data)

        if response.status_code == 200:
            result = response.json()
            indicateurs_phrase = result["indicateurs"][0]
            global indic
            indic = indicateurs_phrase
            indicateurs["TTR"].append(indicateurs_phrase["TTR"])
            indicateurs["taux_adverbes"].append(indicateurs_phrase["adv_rates"])
            indicateurs["taux_verbes_conjugué"].append(indicateurs_phrase["verb_conj_rate"])
            indicateurs["densite_concept_total"].append(indicateurs_phrase["total_concept_density"])
            indicateurs["phrases"].append(phrase)

            emotions["anger_rate"].append(indicateurs_phrase["anger_rate"])
            emotions["disgust_rate"].append(indicateurs_phrase["disgust_rate"])
            emotions["fear_rate"].append(indicateurs_phrase["fear_rate"])
            emotions["joy_rate"].append(indicateurs_phrase["joy_rate"])
            emotions["sadness_rate"].append(indicateurs_phrase["sadness_rate"])
            emotions["surprise_rate"].append(indicateurs_phrase["surprise_rate"])
            emotions["phrases"].append(phrase)

            if len(indicateurs_T["TTR"]) == 0:  
                
                print("godamnit")
                indicateurs_T["TTR"].append(result["indicateurs"][0]["TTR"])
                indicateurs_T["taux_adverbes"].append(result["indicateurs"][0]["adv_rates"])
                indicateurs_T["taux_verbes_conjugué"].append(result["indicateurs"][0]["verb_conj_rate"])
                indicateurs_T["densite_concept_total"].append(result["indicateurs"][0]["total_concept_density"])

                emotions_T["anger_rate"].append(result["indicateurs"][0]["anger_rate"])
                emotions_T["disgust_rate"].append(result["indicateurs"][0]["disgust_rate"])
                emotions_T["fear_rate"].append(result["indicateurs"][0]["fear_rate"])
                emotions_T["joy_rate"].append(result["indicateurs"][0]["joy_rate"])
                emotions_T["sadness_rate"].append(result["indicateurs"][0]["sadness_rate"])
                emotions_T["surprise_rate"].append(result["indicateurs"][0]["surprise_rate"])

                indicateurs_T["phrases"].append(phrase)
                emotions_T["phrases"].append(phrase)

            else:  
                if len(indicateurs["TTR"]) > 1: 
                    indicateurs_T["TTR"].append(indicateurs["TTR"][-1] - indicateurs["TTR"][-2])
                    indicateurs_T["taux_adverbes"].append(indicateurs["taux_adverbes"][-1] - indicateurs["taux_adverbes"][-2])
                    indicateurs_T["taux_verbes_conjugué"].append(indicateurs["taux_verbes_conjugué"][-1] - indicateurs["taux_verbes_conjugué"][-2])
                    indicateurs_T["densite_concept_total"].append(indicateurs["densite_concept_total"][-1] - indicateurs["densite_concept_total"][-2])

                    emotions_T["anger_rate"].append(emotions["anger_rate"][-1] - emotions["anger_rate"][-2])
                    emotions_T["disgust_rate"].append(emotions["disgust_rate"][-1] - emotions["disgust_rate"][-2])
                    emotions_T["fear_rate"].append(emotions["fear_rate"][-1] - emotions["fear_rate"][-2])
                    emotions_T["joy_rate"].append(emotions["joy_rate"][-1] - emotions["joy_rate"][-2])
                    emotions_T["sadness_rate"].append(emotions["sadness_rate"][-1] - emotions["sadness_rate"][-2])
                    emotions_T["surprise_rate"].append(emotions["surprise_rate"][-1] - emotions["surprise_rate"][-2])

                indicateurs_T["phrases"].append(phrase)
                emotions_T["phrases"].append(phrase)

        requete_en_cours = False

        x_data = list(range(len(indicateurs["phrases"])))
        line1.set_data(x_data, indicateurs["TTR"])
        line2.set_data(x_data, indicateurs["taux_adverbes"])
        line3.set_data(x_data, indicateurs["taux_verbes_conjugué"])
        line4.set_data(x_data, indicateurs["densite_concept_total"])

        lineE1.set_data(x_data, emotions["anger_rate"])
        lineE2.set_data(x_data, emotions["disgust_rate"])
        lineE3.set_data(x_data, emotions["fear_rate"])
        lineE4.set_data(x_data, emotions["joy_rate"])
        lineE5.set_data(x_data, emotions["sadness_rate"])
        lineE6.set_data(x_data, emotions["surprise_rate"])

        if len(indicateurs_T["TTR"]) > 0:
            lineT1.set_data(x_data, indicateurs_T["TTR"])
        if len(indicateurs_T["taux_adverbes"]) > 0:
            lineT2.set_data(x_data, indicateurs_T["taux_adverbes"])
        if len(indicateurs_T["taux_verbes_conjugué"]) > 0:
            lineT3.set_data(x_data, indicateurs_T["taux_verbes_conjugué"])
        if len(indicateurs_T["densite_concept_total"]) > 0:
            lineT4.set_data(x_data, indicateurs_T["densite_concept_total"])

        if len(emotions_T["anger_rate"]) > 0:
            lineTE1.set_data(x_data, emotions_T["anger_rate"])
        if len(emotions_T["disgust_rate"]) > 0:
            lineTE2.set_data(x_data, emotions_T["disgust_rate"])
        if len(emotions_T["fear_rate"]) > 0:
            lineTE3.set_data(x_data, emotions_T["fear_rate"])
        if len(emotions_T["joy_rate"]) > 0:
            lineTE4.set_data(x_data, emotions_T["joy_rate"])
        if len(emotions_T["sadness_rate"]) > 0:
            lineTE5.set_data(x_data, emotions_T["sadness_rate"])
        if len(emotions_T["surprise_rate"]) > 0:
            lineTE6.set_data(x_data, emotions_T["surprise_rate"])

        if len(indicateurs['phrases']) > 100:
            ax[0, 0].set_xlim(0, len(indicateurs["phrases"]))
            ax[0, 0].set_ylim(0, max(max(indicateurs["TTR"], default=1), 1))

            ax[0, 1].set_xlim(0, len(emotions["phrases"]))
            ax[0, 1].set_ylim(0, max(max(emotions["anger_rate"], default=0.1), 0.1))

            ax[1, 0].set_xlim(0, len(indicateurs_T["phrases"]))
            ax[1, 0].set_ylim(0, max(max(indicateurs_T["TTR"], default=1), 1))

            ax[1, 1].set_xlim(0, len(emotions_T["phrases"]))
            ax[1, 1].set_ylim(0, max(max(emotions_T["anger_rate"], default=0.1), 0.1))
    return line1, line2, line3, line4, lineE1, lineE2, lineE3, lineE4, lineE5, lineE6, lineT1, lineT2, lineT3, lineT4, lineTE1, lineTE2, lineTE3, lineTE4, lineTE5, lineTE6

ani = FuncAnimation(fig, update, frames=len(phrases), repeat=False)

plt.show()

print('boop')
pdf_filename = "rapport_seances.pdf"
generate_PDF.generate_reports(os.path.join(result_path, "result_indicateurs.json"), pdf_filename)
print(f"Rapport PDF généré: {pdf_filename}")
print("gundamn")
tab = np.array([])
tab2 = np.array([])
for i in indic:
    if(i != 'has_unit' and i != 'ratio_unit'):
        print(i)
        tab = np.append(tab, indic[i])
    print(i)
    tab2 = np.append(tab2, indic[i])

tab = tab.reshape(1, -1)
tab2 = tab2.reshape(1, -1)

adaboostP = adaboost.predict(tab)
grad_boostP = grad_boost.predict(tab)
#randomForestP = randomForest.predict(tab2)
regressionP = regression.predict(tab)
svmP = svm.predict(tab)
votingP = voting.predict(tab)

tab = [adaboostP, grad_boostP, regressionP, svmP, votingP]
print(tab)

import tkinter as tk
from tkinter import ttk

modeles = [
    ("AdaBoost", adaboostP[0]),
    ("Gradient Boosting", grad_boostP[0]),
    ("Régression", regressionP[0]),
    ("SVM", svmP[0]),
    ("Voting Classifier", votingP[0])
]

# Prédiction attendue
prediction_attendue = 0

# Création de la fenêtre
fenetre = tk.Tk()
fenetre.title("Comparaison des prédictions")
fenetre.geometry("500x250")

# Partie gauche : tableau des modèles et de leurs prédictions
frame_gauche = ttk.Frame(fenetre)
frame_gauche.pack(side="left", padx=20, pady=20)

label_gauche = ttk.Label(frame_gauche, text="Prédictions des modèles", font=('Arial', 12, 'bold'))
label_gauche.grid(row=0, column=0, columnspan=2, pady=(0, 10))

ttk.Label(frame_gauche, text="Modèle").grid(row=1, column=0, padx=5, sticky="w")
ttk.Label(frame_gauche, text="Prédiction").grid(row=1, column=1, padx=5, sticky="w")

for i, (nom, prediction) in enumerate(modeles, start=2):
    ttk.Label(frame_gauche, text=nom).grid(row=i, column=0, sticky="w")
    ttk.Label(frame_gauche, text=str(prediction)).grid(row=i, column=1, sticky="w")

# Partie droite : prédiction attendue
frame_droite = ttk.Frame(fenetre)
frame_droite.pack(side="right", padx=20, pady=20)

label_droite = ttk.Label(frame_droite, text="Prédiction attendue", font=('Arial', 12, 'bold'))
label_droite.pack(pady=(0, 10))

valeur = ttk.Label(frame_droite, text=str(prediction_attendue), font=('Arial', 14))
valeur.pack()

fenetre.mainloop()