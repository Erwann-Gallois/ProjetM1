import spacy
import os
import pandas as pd
import csv
from collections import Counter
import numpy as np
import json
from spacy.lang.fr.stop_words import STOP_WORDS
import datetime
from nltk.corpus import wordnet as wn

from pathlib import Path

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
mots_francais = set()
for synset in wn.all_synsets():
    for lemma in synset.lemmas('fra'):  # 'fra' pour le français
        mots_francais.add(lemma.name())
nlp = spacy.load('fr_core_news_lg')
# Supprimer les doublons
#dictionnaire_complet = dictionnaire_complet.drop_duplicates()
chemin_fichier = os.path.join(os.path.dirname(__file__), 'mots_classifies.csv')
# Lire le fichier CSV
dictionnaire_complet = pd.read_csv(chemin_fichier)
# affichage du resultat
print(dictionnaire_complet)

dataset_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr")
result_path = os.path.join(os.path.dirname(__file__), "resultat")
dict_path = os.path.join(os.path.dirname(__file__), "dictionnaire")
AFINN_path = os.path.join(os.path.dirname(__file__), "dictionnaire/AFINN/AFINN-111_FR.txt")
Affin = pd.read_csv(AFINN_path, sep="\t")
Affin.columns = ['word', 'number']
Affin = Affin['word']
NRC_path = os.path.join(os.path.dirname(__file__), "dictionnaire/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/OneFilePerLanguage/French-NRC-EmoLex.txt")
Nrc = pd.read_csv(NRC_path, sep="\t")['French Word']
df_units = pd.concat((Affin, Nrc))

"""def classifier_mots(mots):

    mots_classifies = [] 

    for mot in mots:
        doc = nlp(mot)
        if not doc:
            continue 
        
        mot_classifie = {
            "mot": mot,
            "ACTEURS": "",
            "OBJETS": "",
            "LIEUX": "",
            "ACTIONS": ""
        }


        for token in doc:
            if token.ent_type_ in {"LOC", "GPE", "FAC"}:  # Lieu détecté
                if mot_classifie["LIEUX"] == "":  # Éviter les doublons
                    mot_classifie["LIEUX"] = mot
            elif token.pos_ == "VERB":  # Verbe -> Action
                if mot_classifie["ACTIONS"] == "":
                    mot_classifie["ACTIONS"] = mot
            elif token.pos_ == "PROPN":  # Nom propre -> Acteur
                if mot_classifie["ACTEURS"] == "":
                    mot_classifie["ACTEURS"] = mot
            elif token.pos_ == "NOUN":  # Nom commun -> Objet
                if mot in mots_francais and mot_classifie["OBJETS"] == "":
                    mot_classifie["OBJETS"] = mot
        
        # Ajouter le mot classifié à la liste
        mots_classifies.append(mot_classifie)

    return mots_classifies

def create_uniform_df(classified_words, source_name):
    
    categories = {
        "mot": [],
        "ACTEURS": [],
        "OBJETS": [],
        "LIEUX": [],
        "ACTIONS": []
    }
    for item in classified_words:
        for key in categories:
            categories[key].append(item[key])
    
    # Trouver la longueur maximale parmi toutes les catégories
    max_len = max(len(value) for value in categories.values())

    # Compléter les catégories plus courtes avec des éléments vides (par exemple None)
    for key in categories:
        while len(categories[key]) < max_len:
            categories[key].append(None)

    # Ajouter la colonne de provenance
    categories['provenance'] = [source_name] * max_len

    # Créer le DataFrame avec toutes les colonnes de la même longueur
    df = pd.DataFrame(categories)
    return df


def ajouter_provenance(df, source):
    df['provenance'] = source
    return df


print(1)
tab1 = classifier_mots(mots_francais)  # Listes de mots français
print(2)
tab2 = classifier_mots(Nrc)  # Liste de mots de NRC
print(3)
tab3 = classifier_mots(Affin)  # Liste de mots de AFINN
print("fini")

# Supprimer les duplicata dans chaque catégorie en utilisant un set
df_tab1 = create_uniform_df(tab1, "mots_francais")
df_tab2 = create_uniform_df(tab2, "NRC")
df_tab3 = create_uniform_df(tab3, "AFINN")

print(df_tab1.shape)
print(df_tab2.shape)
print(df_tab3.shape)

# Convertir chaque catégorie en DataFrame
df_tab1 = pd.DataFrame({key: value for key, value in df_tab1.items()})
df_tab2 = pd.DataFrame({key: value for key, value in df_tab2.items()})
df_tab3 = pd.DataFrame({key: value for key, value in df_tab3.items()})

dictionnaire_complet = pd.concat([df_tab1, df_tab2, df_tab3], ignore_index=True, sort=False)
print(dictionnaire_complet)
dictionnaire_complet.to_csv('mots_classifies.csv', index=False, encoding='utf-8')

print("Le fichier CSV a été généré avec succès.")"""

def stats_words(texte):
    doc = nlp(texte)
    freq_list = Counter(token.text for token in doc if not token.is_punct and not token.pos_ == "DET" and not token.is_stop and not "\n"  in token.text)
    mean = np.mean(list(freq_list.values()))
    range = max(freq_list.values()) - min(freq_list.values())
    std = np.std(list(freq_list.values()))
    return mean, range, std

def lexical_diversity(texte, a):
    doc = nlp(texte)
    freq_list = Counter(token.text for token in doc if not token.is_punct and not token.is_stop and not "\n" in token.text)
    N = len(doc)
    V = len(freq_list)
    v1 = len([word for word in freq_list if freq_list[word] == 1])
    brunet_index = N ** (V ** -a)
    honore_stats = 100 * np.log(N) / (1 - v1 / N)
    ratio_unique = v1 / N
    return brunet_index, honore_stats, ratio_unique

def emotionnal_analysis(texte):
    file_path = os.path.join(dict_path, "FEEL.csv")
    emotion_df = pd.read_csv(file_path, sep=";", encoding="utf-8")
    emotion_df = emotion_df.drop(columns=["id", "polarity"]) 

    doc = nlp(texte)
    freq_list = Counter(token.text.lower() for token in doc if not token.is_punct and not token.is_stop)

    emotion_dict = {"joy": 0, "fear": 0, "sadness": 0, "anger": 0, "surprise": 0, "disgust": 0}

    for word in freq_list:
        if word in emotion_df["word"].values:
            emotion_row = emotion_df[emotion_df["word"] == word].iloc[0]
            for emotion in emotion_dict.keys():
                emotion_dict[emotion] += freq_list[word] * emotion_row[emotion]
    
    return emotion_dict

def positif_negatif(texte):
    afinn = {}
    with open(os.path.join(dict_path, "AFINN-111_translated.txt"), encoding="utf-8") as f:
        for line in f:
            k, v = line.strip().split('\t')
            afinn[k] = int(v)

    doc = nlp(texte)
    score = 0
    for sent in doc.sents:  
        sent_text = str(sent).lower()  
        score += sum(afinn.get(word, 0) for word in sent_text.split())

    return score

def stats_morpho(texte):
    doc = nlp(texte)
   
    # POS tags
    pos_counts = {
        "ADJ": 0,  # Adjectifs
        "ADP": 0,  # Adpositions (prepositions/postpositions)
        "ADV": 0,  # Adverbes
        "CONJ": 0,  # Conjunctions
        "DET": 0,  # Determinants
        "NOUN": 0,  # Noms
        "PRON": 0,  # Pronoms
        "VERB": 0,  # Verbes
        "PROPN": 0,  # Noms propres
        "AUX": 0,   # Auxiliaires
    }

    conjug_count = 0
    inf_count = 0

    total_count = 0

    verb_obj = 0
    verb_suj = 0
    verb_aux = 0
    total_verbs = 0

    repetition_cons = 0
    prop_sub = 0
    total_phrase = 0

    for phrase in doc.sents:
        total_phrase += 1
        contient_vo = False
        contient_vs = False
        contient_va = False
        token_list = set()

        for token in phrase:
            if not token.is_punct and token.text != "\n":
                total_count += 1

                if token.lemma_ not in STOP_WORDS:
                    if token.lemma_ in token_list:
                        repetition_cons += 1
                    else:
                        token_list.add(token.lemma_)

                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1
                if token.pos_ == "VERB":
                    total_verbs += 1
                    
                    if token.morph.get("VerbForm"):
                        if token.morph.get("VerbForm")[0] == "Inf":
                            inf_count += 1
                        else:
                            conjug_count += 1

                    if any(child.dep_ in {"obj", "iobj"} for child in token.children):
                        contient_vo = True

                    if any(child.dep_ in {"nsubj", "csubj"} for child in token.children):
                        contient_vs = True

                    if any(child.pos_ == "AUX" for child in token.children):
                        contient_va = True

            if token.dep_ in {"advcl", "csubj", "ccomp", "relcl"}:
                prop_sub += 1

        if contient_vo:
            verb_obj += 1
        if contient_vs:
            verb_suj += 1
        if contient_va:
            verb_aux += 1

    pos_rates = {pos: count / total_count for pos, count in pos_counts.items()}

    mean_prop_sub = prop_sub / total_phrase

    nb_ver = pos_counts["VERB"]
    if(nb_ver > 0):
        rate_conjug = conjug_count/nb_ver
        rate_inf = inf_count/nb_ver
    else:
        rate_conjug = 0
        rate_inf = 0
    pos_rates = {pos: count / total_count for pos, count in pos_counts.items()}
    verb_w_obj = verb_obj / total_phrase
    verb_w_subj = verb_suj / total_phrase
    verb_w_aux = verb_aux / total_phrase
    return pos_rates, total_count, rate_conjug, rate_inf, verb_w_obj, verb_w_subj, verb_w_aux, repetition_cons, mean_prop_sub

from collections import Counter
import spacy

# Assurez-vous que vous avez chargé un modèle spacy adapté
nlp = spacy.load("fr_core_news_sm")  # Exemple pour le français

def unit_analysis(texte, timeDiff):
    doc = nlp(texte)

    # Initialisation des dictionnaires et des compteurs
    unit_ratio = {}
    unit_count = Counter()  # Compter les unités d'information par catégorie
    unit_mention = set()  # Ensemble des unités mentionnées au moins une fois
    
    # Créer un set de toutes les unités d'information (en minuscules) et leur catégorie
    units_set = set(dictionnaire_complet['mot'].str.lower())  # Unités d'information (en minuscules)
    categories = dictionnaire_complet.columns[1:-1]  # Exclure la première et la dernière colonne ('mot' et 'provenance')
    
    # Total des mots dans le texte
    total_words = len([token for token in doc if not token.is_punct and not token.is_space])

    # Créer un dictionnaire pour stocker les comptages par catégorie
    category_counts = {category: Counter() for category in categories}

    # Parcourir les tokens du texte
    for token in doc:
        # Si le lemme du token est une unité d'information
        lemma = token.lemma_.lower()
        if lemma in units_set:
            # Trouver les catégories de l'unité d'information
            for category in categories:
                if pd.notna(dictionnaire_complet[dictionnaire_complet['mot'].str.lower() == lemma][category].values[0]):
                    # Incrémenter le compteur de cette unité dans la catégorie correspondante
                    category_counts[category][lemma] += 1
                    unit_mention.add(lemma)  # Ajouter l'unité à celles mentionnées
    
    # Calculer les ratios des unités d'information par rapport au total des mots
    unit_ratio = {unit: count / total_words for category in category_counts for unit, count in category_counts[category].items()}
    
    # Densité des concepts uniques : unités mentionnées au moins une fois / total des mots
    unique_concept_density = len(unit_mention) / total_words if total_words > 0 else 0
    
    # Efficacité des concepts uniques : unités mentionnées au moins une fois / durée (en secondes)
    unique_concept_efficiency = len(unit_mention) / timeDiff if timeDiff > 0 else 0
    
    # Densité totale des concepts : total des mots faisant référence à des unités / total des mots
    total_concept_density = sum(sum(category_counts[category].values()) for category in category_counts) / total_words if total_words > 0 else 0
    
    # Efficacité totale des concepts : total des mots faisant référence à des unités / durée (en secondes)
    total_concept_efficiency = sum(sum(category_counts[category].values()) for category in category_counts) / timeDiff if timeDiff > 0 else 0
    
    # Retourner les résultats sous forme de dictionnaire
    has_unit = {category: dict(category_counts[category]) for category in category_counts}
    
    return has_unit, unit_ratio, unique_concept_efficiency, unique_concept_density, total_concept_density, total_concept_efficiency


def export_patient_dialogue(file_path):
    try :
        df = pd.read_csv(file_path, sep="\t")
    except :
        print("File not found")
        return None
    df = pd.read_csv(file_path, sep="\t")
    patient_df = df[df["speaker"] == 2].reset_index()
    patient_df = patient_df.drop(columns=["index", "original", "speaker"])
    patient_tab = patient_df.to_numpy()
    str_dialogue = " ".join(patient_tab[:, 0])
    return str_dialogue


def stats_morpho_all(patient_dialogue, timeDiff):
    pos_rates, total_word, rate_conj, rate_inf, verb_w_obj, verb_w_subj, verb_w_aux, repetition_cons, mean_prop_sub = stats_morpho(patient_dialogue)
    mean, range, std = stats_words(patient_dialogue)
    brunet_index, honore_stats, ratio_unique = lexical_diversity(patient_dialogue, 0.165)
    emotion_dict = emotionnal_analysis(patient_dialogue)
    score = positif_negatif(patient_dialogue)
    has_unit, ratio_unit, unique_concept_efficiency, unique_concept_density, total_concept_density, total_concept_efficiency = unit_analysis(patient_dialogue, timeDiff)
    json_file = {
        "adj_rate" : pos_rates["ADJ"],
        "adp_rate" : pos_rates["ADP"],
        "adv_rates": pos_rates["ADV"],  
        "conj_rate": pos_rates["CONJ"],  
        "det_rate": pos_rates["DET"],  
        "noun_rate": pos_rates["NOUN"],
        "pron_rate": pos_rates["PRON"],
        "verb_rate": pos_rates["VERB"],
        "propn_rate": pos_rates["PROPN"],
        "verb_aux_rate": verb_w_aux,
        "verb_obj_rate": verb_w_obj,
        "verb_subj_rate": verb_w_subj,
        "sub_prop_rate": mean_prop_sub,
        "repetition_cons_rate": repetition_cons,
        "verb_conj_rate" : rate_conj,
        "verb_inf_rate" : rate_inf,
        "total_words" : total_word,
        "mean_freq_words" : float(mean),
        "range_freq_words" : range,
        "std_freq_words" : float(std),
        "Brunet_index" : brunet_index,
        "Honore_statistic" : float(honore_stats),
        "TTR" : ratio_unique, 
        "anger_rate" : emotion_dict["anger"]/total_word,
        "disgust_rate" : emotion_dict["disgust"]/total_word,
        "fear_rate" : emotion_dict["fear"]/total_word,
        "joy_rate" : emotion_dict["joy"]/total_word,
        "sadness_rate" : emotion_dict["sadness"]/total_word,
        "surprise_rate" : emotion_dict["surprise"]/total_word,
        "Score_AFINN" : score,
        "has_unit": has_unit,
        "ratio_unit": ratio_unit,
        "unique_concept_density": unique_concept_density,
        "unique_concept_efficiency": unique_concept_efficiency,
        "total_concept_density": total_concept_density,
        "total_concept_efficiency": total_concept_efficiency
    }
    return json_file


# file_path = "DAMT_FR/FR_D0420-S1-T05.csv"
# timeDiff = int(datetime.timedelta(seconds = 360).total_seconds())
# file = stats_morpho_all(export_patient_dialogue(os.path.join(dataset_path, file_path)), file_path.split("/")[-1].split(".")[0], timeDiff)

aggregated_results = {
        "adj_rate": [],
        "adp_rate": [],
        "adv_rates": [],
        "conj_rate": [],
        "det_rate": [],
        "noun_rate": [],
        "pron_rate": [],
        "verb_rate": [],
        "propn_rate": [],
        "verb_aux_rate": [],
        "verb_obj_rate": [],
        "verb_subj_rate": [],
        "sub_prop_rate": [],
        "repetition_cons_rate": [],
        "verb_conj_rate": [],
        "verb_inf_rate": [],
        "total_words": [],
        "mean_freq_words": [],
        "range_freq_words": [],
        "std_freq_words": [],
        "Brunet_index": [],
        "Honore_statistic": [],
        "TTR": [],
        "anger_rate": [],
        "disgust_rate": [],
        "fear_rate": [],
        "joy_rate": [],
        "sadness_rate": [],
        "surprise_rate": [],
        "Score_AFINN": [],
        "has_unit": [],
        "ratio_unit": [],
        "unique_concept_density": [],
        "unique_concept_efficiency": [],
        "total_concept_density": [],
        "total_concept_efficiency": [],
        "depressed_binary": []
    }
        
        

def segmenter_phrases(texte):
    doc = nlp(texte)
    return [phrase.text.strip() for phrase in doc.sents]

train_path = os.path.join(os.path.dirname(__file__), "data_model/train.json")
test_path = os.path.join(os.path.dirname(__file__), "data_model/test.json")
validation_path = os.path.join(os.path.dirname(__file__), "data_model/validation.json")

# Read the data files
data1 = pd.read_json(train_path, lines=True)
data2 = pd.read_json(test_path, lines=True)
data3 = pd.read_json(validation_path, lines=True)

timeDiff = 60*60
diversity_a = 0.5

combined_data = pd.concat([data1, data2, data3])

"""for i in combined_data.index:
    dat = combined_data.iloc[i]
    phrases = segmenter_phrases("".join(dat['dialogue']))  # Assuming this returns a list of phrases
    phrase = "".join(phrases)  # Concatenate the list of phrases into one string
    pos_rates, total_words_count, verb_conjug_rate, verb_inf_rate, verb_obj_rate, verb_subj_rate, verb_aux_rate, repetition_cons_count, mean_subordination = stats_morpho(phrase)
    aggregated_results["adj_rate"].append(pos_rates.get("ADJ", 0))
    aggregated_results["adp_rate"].append(pos_rates.get("ADP", 0))
    aggregated_results["adv_rates"].append(pos_rates.get("ADV", 0))
    aggregated_results["conj_rate"].append(pos_rates.get("CONJ", 0))
    aggregated_results["det_rate"].append(pos_rates.get("DET", 0))
    aggregated_results["noun_rate"].append(pos_rates.get("NOUN", 0))
    aggregated_results["pron_rate"].append(pos_rates.get("PRON", 0))
    aggregated_results["verb_rate"].append(pos_rates.get("VERB", 0))
    aggregated_results["propn_rate"].append(pos_rates.get("PROPN", 0))
    aggregated_results["verb_aux_rate"].append(verb_aux_rate)
    aggregated_results["verb_obj_rate"].append(verb_obj_rate)
    aggregated_results["verb_subj_rate"].append(verb_subj_rate)
    aggregated_results["sub_prop_rate"].append(mean_subordination)
    aggregated_results["repetition_cons_rate"].append(repetition_cons_count)
    aggregated_results["verb_conj_rate"].append(verb_conjug_rate)
    aggregated_results["verb_inf_rate"].append(verb_inf_rate)
    aggregated_results["total_words"].append(total_words_count)
    
    moyenne_freq, plage_freq, ecart_type_freq = stats_words(phrase)
    aggregated_results["mean_freq_words"].append(moyenne_freq)
    aggregated_results["range_freq_words"].append(plage_freq)
    aggregated_results["std_freq_words"].append(ecart_type_freq)
    
    brunet_index, honore_statistic, ttr_value = lexical_diversity(phrase, diversity_a)
    aggregated_results["Brunet_index"].append(brunet_index)
    aggregated_results["Honore_statistic"].append(honore_statistic)
    aggregated_results["TTR"].append(ttr_value)
    
    emotion_values = emotionnal_analysis(phrase)
    aggregated_results["anger_rate"].append(emotion_values.get("anger", 0))
    aggregated_results["disgust_rate"].append(emotion_values.get("disgust", 0))
    aggregated_results["fear_rate"].append(emotion_values.get("fear", 0))
    aggregated_results["joy_rate"].append(emotion_values.get("joy", 0))
    aggregated_results["sadness_rate"].append(emotion_values.get("sadness", 0))
    aggregated_results["surprise_rate"].append(emotion_values.get("surprise", 0))
    
    score_afinn = positif_negatif(phrase)
    aggregated_results["Score_AFINN"].append(score_afinn)
    
    unit_has, unit_ratio, unique_concept_efficiency_value, unique_concept_density_value, total_concept_density_value, total_concept_efficiency_value = unit_analysis(phrase, timeDiff)
    aggregated_results["has_unit"].append(unit_has)
    aggregated_results["ratio_unit"].append(unit_ratio)
    aggregated_results["unique_concept_density"].append(unique_concept_density_value)
    aggregated_results["unique_concept_efficiency"].append(unique_concept_efficiency_value)
    aggregated_results["total_concept_density"].append(total_concept_density_value)
    aggregated_results["total_concept_efficiency"].append(total_concept_efficiency_value)
    aggregated_results["depressed_binary"].append(dat["PHQ_Binary"])
    print(dat["Participant_ID"])

with open(os.path.join(result_path, "result_data.json"), "w", encoding="utf-8") as fichier_sortie:
    json.dump(aggregated_results, fichier_sortie, indent=4,
            default=lambda objet: objet.item() if hasattr(objet, "item") else objet)
print("Fichier json généré")
    """