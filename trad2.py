import pandas as pd
import os
import psycopg2
import json
from transformers import MarianMTModel, MarianTokenizer


# 📌 Définition de la clé API pour Gemini
model_name = "Helsinki-NLP/opus-mt-en-fr"

# 📌 Dossier contenant les fichiers
data_path = os.path.join(os.path.dirname(__file__), "data_model")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
# 📌 Associer chaque fichier à une valeur pour dataset_type
file_mapping = {
    "test.json": 1,
    "validation.json": 2,
    "train.json": 3
}

# 📌 Connexion PostgreSQL
connexion = psycopg2.connect(
    dbname="textes_db",
    user="utilisateur",
    password="mdp",
    host="localhost",
    port="5432"
)
curseur = connexion.cursor()

# 📌 Fonction de traduction avec Gemini API
def translate_text(texts):
    text_fr = []
    for text in texts:
        tokenized_text = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**tokenized_text)
        text_fr.append(tokenizer.decode(translated[0], skip_special_tokens=True))
    return text_fr
    

# 📌 Parcourir les fichiers (test, validation, train)
for file_name, dataset_type in file_mapping.items():
    file_path = os.path.join(data_path, file_name)
    # Vérifier si le fichier existe
    if not os.path.exists(file_path):
        print(f"⚠️ Fichier {file_name} non trouvé, passage au suivant.")
        continue

    # Lire ligne par ligne (JSONL)
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))  # Supprime les espaces blancs
            except json.JSONDecodeError as e:
                print(f"⚠️ Erreur de décodage JSON à la ligne : {line}\nErreur : {e}")
    print(f"✅ Chargement réussi de {len(data)} participants depuis {file_name}.")

    # 📌 Insérer les données dans PostgreSQL avec traduction
    for participant in data:
        participant_id = participant["Participant_ID"]
        gender = participant["Gender"]
        phq_binary = participant["PHQ_Binary"]
        phq_score = participant["PHQ_Score"]
        dialogue = participant["dialogue"]

        # Traduire le dialogue
        translated_dialogue = translate_text(dialogue)
        # Insérer dans PostgreSQL
        curseur.execute("""
            INSERT INTO participants (
                Participant_ID, Gender, PHQ_Binary, PHQ_Score, dialogue, dialogue_fr,
                PHQ8_1_NoInterest, PHQ8_2_Depressed, PHQ8_3_Sleep, PHQ8_4_Tired,
                PHQ8_5_Appetite, PHQ8_6_Failure, PHQ8_7_Concentration, PHQ8_8_Psychomotor,
                partof, indicateur
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            participant_id,
            gender,
            phq_binary,
            phq_score,
            json.dumps(dialogue),          # Convertir dialogue en JSONB
            json.dumps(translated_dialogue), # Stocker la traduction
            participant.get("PHQ8_1_NoInterest", None),
            participant.get("PHQ8_2_Depressed", None),
            participant.get("PHQ8_3_Sleep", None),
            participant.get("PHQ8_4_Tired", None),
            participant.get("PHQ8_5_Appetite", None),
            participant.get("PHQ8_6_Failure", None),
            participant.get("PHQ8_7_Concentration", None),
            participant.get("PHQ8_8_Psychomotor", None),
            dataset_type,  # Indiquer la provenance du fichier
            participant.get("indicateur", None)  # Ajouter `indicateur`
        ))
        connexion.commit()
        print(f"✅ Données insérées pour le participant {participant_id}.")

connexion.close()
print("✅ Données insérées et traduites avec succès dans la base de données !")
