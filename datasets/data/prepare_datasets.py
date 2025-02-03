from datasets import DatasetDict, Dataset
import json
import os
import csv
import re


# ----- DAMT
def prepare_damt_transcript(transcript):
    conversation = ""
    for turn in transcript:
        speaker = "Doctor" if turn["speaker"] == 1 else "Patient"
        dialogue = " ".join(turn["dialogue"])
        conversation += f"{speaker}: {dialogue}\n"

    return conversation


def prepare_damt_casenote(transcript, note, include_source):
    categories = [
        "Client Details",
        "Chief Complaint",
        "History of Present Illness",
        "Past Psychiatric History",
        "History of Substance Use",
        "Social History",
        "Family History",
        "Review of Systems",
    ]
    casenote_data = {category: [] for category in categories}

    transformed_transcript = []
    for entry in transcript:
        speaker = entry["speaker"]
        dialogues = entry["dialogue"]

        for sentence in dialogues:
            transformed_transcript.append({"speaker": speaker, "dialogue": sentence})

    for element in note:
        category_id = int(element["categoryId"])
        category = categories[category_id]

        source_sentence_id = int(element["sourceId"])
        source_sentence = transformed_transcript[source_sentence_id]["dialogue"]

        formal_text = element["formalText"]
        casenote_text = (
            f'"{source_sentence}" => {formal_text}' if include_source else formal_text
        )

        casenote_data[category].append(casenote_text)

    casenote = ""
    for category in casenote_data.keys():
        if len(casenote_data[category]) > 0:
            casenote += f"{category}:\n\n"
            casenote += "\n".join(casenote_data[category])
            casenote += "\n\n"

    return casenote


def prepare_damt_dataset(dataset_path, annotator=1, include_source=False):
    transcripts_path = f"{dataset_path}/transcripts/transcribed/".replace("//", "/")
    casenotes_path = f"{dataset_path}/casenotes/".replace("//", "/")
    file_names = os.listdir(casenotes_path + f"annotator_{annotator}")

    data = []
    for file_name in file_names:
        element = {}

        # Transcript prep
        transcript_file = open(transcripts_path + file_name, "r")
        transcript = json.load(transcript_file)
        conversation = prepare_damt_transcript(transcript)
        element["dialogue"] = conversation

        # Casenote prep
        casenote_file = open(f"{casenotes_path}annotator_{annotator}/{file_name}", "r")
        casenote = json.load(casenote_file)
        casenote = prepare_damt_casenote(transcript, casenote, include_source)
        element["note"] = casenote

        data.append(element)

    ds = Dataset.from_list(data)
    ds = ds.train_test_split(test_size=0.5, seed=42)

    return ds


# ----- DAIC-WOZ
def prepare_daic_woz(dataset_path):
    files = os.listdir(dataset_path)

    ellie_regex = r"\((.*?)\)"

    raw_dataset = {"dialogue": []}

    for f in files:
        if f[-4:] == ".csv":
            with open(dataset_path + "/" + f, "r".replace("//", "/")) as file:
                csv_reader = csv.reader(file)

                conversation = ""
                for row in csv_reader:
                    if len(row) > 0:
                        convo_turn = row[0].split("\t")
                        speaker, content = convo_turn[2], convo_turn[3]

                        if content == "<sync>":
                            continue

                        if speaker == "Ellie":
                            if re.search(ellie_regex, content):
                                ellie_speech = re.search(ellie_regex, content).group(1)
                                conversation += "Doctor: " + ellie_speech + "\n"
                            else:
                                conversation += "Doctor: " + content + ".\n"

                        if speaker == "Participant":
                            conversation += "Patient: " + content + ".\n"

        raw_dataset["dialogue"].append(conversation)

    daic_woz_dataset = Dataset.from_dict(raw_dataset)
    return daic_woz_dataset


def prepare_memo_dataset(directory_path: str):
    dataset = DatasetDict()

    splits = {
        "train": f"{directory_path}/Train",
        "valid": f"{directory_path}/Validation",
        "test": f"{directory_path}/Test",
    }

    for split in splits:
        data = []
        for filename in os.listdir(splits[split]):
            if filename.endswith(".csv"):
                file_path = os.path.join(splits[split], filename)

                with open(file_path, "r") as file:
                    csv_reader = csv.reader(file)
                    data_item = {}

                    next(csv_reader)

                    dialogue = ""
                    for row in csv_reader:
                        utterance, sub_topic, id, type_, dialogue_act, emotion = row

                        if (
                            utterance == "summary"
                            or utterance == "Summary"
                            or utterance == "summary "
                        ):
                            data_item["note"] = sub_topic
                            continue

                        if (
                            utterance == "primary_topic"
                            or utterance == "secondary_topic"
                        ):
                            continue

                        role = "Patient" if type_ == "P" else "Doctor"
                        dialogue += f"{role}: {utterance}\n"

                    assert data_item["note"] is not None
                    data_item["dialogue"] = dialogue
                    data.append(data_item)

        dataset[split] = Dataset.from_list(data)

    return dataset


# def prepare_hope_dataset(directory_path: str):
#     dataset = DatasetDict()

#     splits = {
#         "train": f"{directory_path}/Train",
#         "valid": f"{directory_path}/Validation",
#         "test": f"{directory_path}/Test",
#     }

#     for split in splits:
#         data = []
#         for filename in os.listdir(splits[split]):
#             if filename.endswith(".csv"):
#                 file_path = os.path.join(splits[split], filename)
#                 print(filename)

#                 with open(file_path, "r") as file:
#                     csv_reader = csv.reader(file, delimiter=",")

#                     data_item = {}

#                     next(csv_reader)

#                     dialogue = ""
#                     for row in csv_reader:
#                         print(row)
#                         (
#                             unnamed_1,
#                             id,
#                             unnamed_2,
#                             id_2,
#                             id_3,
#                             type,
#                             utterance,
#                             dialogue_act_1,
#                             dialogue_act_2,
#                             emotion,
#                         ) = row

#                         if type == "P":
#                             dialogue += f"Patient: {utterance}\n"

#                         if type == "P":
#                             dialogue += f"Patient: {utterance}\n"

#         data_item["dialogue"] = dialogue
#         data.append(data_item)

#         dataset[split] = Dataset.from_list(data)

#     return dataset


# prepare_hope_dataset("./data/HOPE")
