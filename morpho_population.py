import spacy
import os
import pandas as pd
from collections import Counter
import numpy as np
import json
from spacy.lang.fr.stop_words import STOP_WORDS
import datetime

nlp = spacy.load('fr_core_news_lg')
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

def export_patient_dialogue(file_path):
    try:
        df_patient = pd.read_csv(file_path, sep="\t")
    except:
        print("File not found :", file_path)
        return None
    df_patient = df_patient[df_patient["speaker"] == 2].reset_index(drop=True)
    patient_array = df_patient.drop(columns=["original", "speaker"]).to_numpy()
    dialogue_text = " ".join(patient_array[:, 0])
    return dialogue_text

def stats_words(texte):
    doc = nlp(texte)
    freq_list = Counter(token.text for token in doc if not token.is_punct and token.pos_ != "DET" and not token.is_stop and "\n" not in token.text)
    moyenne_frequence = np.mean(list(freq_list.values()))
    plage_frequence = max(freq_list.values()) - min(freq_list.values())
    ecart_type_frequence = np.std(list(freq_list.values()))
    return moyenne_frequence, plage_frequence, ecart_type_frequence

def lexical_diversity(texte, a):
    doc = nlp(texte)
    freq_list = Counter(token.text for token in doc if not token.is_punct and not token.is_stop and "\n" not in token.text)
    nombre_mots = len(doc)
    nombre_vocabulaire = len(freq_list)
    mots_ponctuels = len([mot for mot in freq_list if freq_list[mot] == 1])
    brunet_index = nombre_mots ** (nombre_vocabulaire ** -a)
    honore_statistic = 100 * np.log(nombre_mots) / (1 - mots_ponctuels / nombre_mots)
    ratio_unique = mots_ponctuels / nombre_mots
    return brunet_index, honore_statistic, ratio_unique

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
    with open(os.path.join(dict_path, "AFINN-111_translated.txt"), encoding="utf-8") as fichier:
        for ligne in fichier:
            cle, valeur = ligne.strip().split('\t')
            afinn[cle] = int(valeur)
    doc = nlp(texte)
    score = 0
    for phrase in doc.sents:  
        texte_phrase = str(phrase).lower()  
        score += sum(afinn.get(mot, 0) for mot in texte_phrase.split())
    return score

def stats_morpho(texte):
    doc = nlp(texte)
    pos_counts = {
        "ADJ": 0,
        "ADP": 0,
        "ADV": 0,
        "CONJ": 0,
        "DET": 0,
        "NOUN": 0,
        "PRON": 0,
        "VERB": 0,
        "PROPN": 0,
        "AUX": 0,
    }
    conjug_count = 0
    inf_count = 0
    total_token_count = 0
    verb_with_obj_count = 0
    verb_with_subj_count = 0
    verb_with_aux_count = 0
    total_phrase_count = 0
    repetition_count = 0
    subordination_count = 0

    for phrase in doc.sents:
        total_phrase_count += 1
        contient_objet = False
        contient_sujet = False
        contient_auxiliaire = False
        token_lemmes_utilises = set()
        for token in phrase:
            if not token.is_punct and token.text != "\n":
                total_token_count += 1
                if token.lemma_ not in STOP_WORDS:
                    if token.lemma_ in token_lemmes_utilises:
                        repetition_count += 1
                    else:
                        token_lemmes_utilises.add(token.lemma_)
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1
                if token.pos_ == "VERB":
                    if token.morph.get("VerbForm"):
                        if token.morph.get("VerbForm")[0] == "Inf":
                            inf_count += 1
                        else:
                            conjug_count += 1
                    if any(enfant.dep_ in {"obj", "iobj"} for enfant in token.children):
                        contient_objet = True
                    if any(enfant.dep_ in {"nsubj", "csubj"} for enfant in token.children):
                        contient_sujet = True
                    if any(enfant.pos_ == "AUX" for enfant in token.children):
                        contient_auxiliaire = True
            if token.dep_ in {"advcl", "csubj", "ccomp", "relcl"}:
                subordination_count += 1
        if contient_objet:
            verb_with_obj_count += 1
        if contient_sujet:
            verb_with_subj_count += 1
        if contient_auxiliaire:
            verb_with_aux_count += 1

    pos_rates = {pos: (nombre / total_token_count) for pos, nombre in pos_counts.items()}
    moyenne_subordination = subordination_count / total_phrase_count
    nombre_verbes = pos_counts["VERB"]
    if nombre_verbes > 0:
        rate_conjug = conjug_count / nombre_verbes
        rate_inf = inf_count / nombre_verbes
    else:
        rate_conjug = 0
        rate_inf = 0
    return pos_rates, total_token_count, rate_conjug, rate_inf, verb_with_obj_count / total_phrase_count, verb_with_subj_count / total_phrase_count, verb_with_aux_count / total_phrase_count, repetition_count, moyenne_subordination

def unit_analysis(texte, timeDiff):
    doc = nlp(texte)
    unit_count = Counter()
    units_set = set(df_units.str.lower())
    total_non_punct_tokens = len([token for token in doc if not token.is_punct and not token.is_space])
    for token in doc:
        if token.lemma_.lower() in units_set:
            unit_count[token.lemma_.lower()] += 1
    unit_ratio = {unit: (compte / total_non_punct_tokens) for unit, compte in unit_count.items()}
    unique_concept_density = len(unit_count) / total_non_punct_tokens if total_non_punct_tokens > 0 else 0
    unique_concept_efficiency = len(unit_count) / timeDiff
    total_concept_density = sum(unit_count.values()) / total_non_punct_tokens if total_non_punct_tokens > 0 else 0
    total_concept_efficiency = sum(unit_count.values()) / timeDiff
    return dict(unit_count), unit_ratio, unique_concept_efficiency, unique_concept_density, total_concept_density, total_concept_efficiency

def stats_pop(dataset, timeDiff, number_of_files=None, diversity_a=0.5):
    dataset_directory = os.path.join(dataset_path, dataset)
    list_filenames = os.listdir(dataset_directory)
    if number_of_files is not None and number_of_files < len(list_filenames):
        list_filenames = list_filenames[:number_of_files]
    else:
        print("Le nombre de fichiers demandé est supérieur au nombre de fichiers disponibles.")
    
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
        "total_concept_efficiency": []
    }
    
    for filename in list_filenames:
        print(filename)
        file_path = os.path.join(dataset_directory, filename)
        dialogue_text = export_patient_dialogue(file_path)
        if dialogue_text is None:
            continue
        
        pos_rates, total_words_count, verb_conjug_rate, verb_inf_rate, verb_obj_rate, verb_subj_rate, verb_aux_rate, repetition_cons_count, mean_subordination = stats_morpho(dialogue_text)
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
        
        moyenne_freq, plage_freq, ecart_type_freq = stats_words(dialogue_text)
        aggregated_results["mean_freq_words"].append(moyenne_freq)
        aggregated_results["range_freq_words"].append(plage_freq)
        aggregated_results["std_freq_words"].append(ecart_type_freq)
        
        brunet_index, honore_statistic, ttr_value = lexical_diversity(dialogue_text, diversity_a)
        aggregated_results["Brunet_index"].append(brunet_index)
        aggregated_results["Honore_statistic"].append(honore_statistic)
        aggregated_results["TTR"].append(ttr_value)
        
        emotion_values = emotionnal_analysis(dialogue_text)
        aggregated_results["anger_rate"].append(emotion_values.get("anger", 0))
        aggregated_results["disgust_rate"].append(emotion_values.get("disgust", 0))
        aggregated_results["fear_rate"].append(emotion_values.get("fear", 0))
        aggregated_results["joy_rate"].append(emotion_values.get("joy", 0))
        aggregated_results["sadness_rate"].append(emotion_values.get("sadness", 0))
        aggregated_results["surprise_rate"].append(emotion_values.get("surprise", 0))
        
        score_afinn = positif_negatif(dialogue_text)
        aggregated_results["Score_AFINN"].append(score_afinn)
        
        unit_has, unit_ratio, unique_concept_efficiency_value, unique_concept_density_value, total_concept_density_value, total_concept_efficiency_value = unit_analysis(dialogue_text, timeDiff)
        aggregated_results["has_unit"].append(unit_has)
        aggregated_results["ratio_unit"].append(unit_ratio)
        aggregated_results["unique_concept_density"].append(unique_concept_density_value)
        aggregated_results["unique_concept_efficiency"].append(unique_concept_efficiency_value)
        aggregated_results["total_concept_density"].append(total_concept_density_value)
        aggregated_results["total_concept_efficiency"].append(total_concept_efficiency_value)
    
    with open(os.path.join(result_path, "result_" + dataset + ".json"), "w", encoding="utf-8") as fichier_sortie:
        json.dump(aggregated_results, fichier_sortie, indent=4,
                default=lambda objet: objet.item() if hasattr(objet, "item") else objet)
    print("Fichier json généré")

timeDiff = int(datetime.timedelta(seconds=360).total_seconds())

"""
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

diversity_a = 0.5

train_path = os.path.join(os.path.dirname(__file__), "data_model/train.json")
test_path = os.path.join(os.path.dirname(__file__), "data_model/test.json")
validation_path = os.path.join(os.path.dirname(__file__), "data_model/validation.json")

# Read the data files
data1 = pd.read_json(train_path, lines=True)
data2 = pd.read_json(test_path, lines=True)
data3 = pd.read_json(validation_path, lines=True)

combined_data = pd.concat([data1, data2, data3])

for i in combined_data.index:
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
    
data = os.path.join(os.path.dirname(__file__), "resultat/result_data.json")
df = pd.read_json(data)
df = df.select_dtypes(include=['number'])
print(df)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.show()

df["depressed_binary"] = df["depressed_binary"].astype(str)

for col in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["depressed_binary"], y=df[col])
    plt.title(f"Boxplot de {col} selon {"depressed_binary"}")
    plt.xlabel("depressed_binary")
    plt.ylabel(col)
    plt.show()