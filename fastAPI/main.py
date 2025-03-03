from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import morpho
import datetime


app = FastAPI()

class Text(BaseModel):
    texte: str = "" 
    phrase: Optional[List[str]] = []
    indicateurs: Optional[List[str]] = []

resultat_path = Path(os.getcwd()).parent / "resultat" / "indicateurs.json"

resultat_path.parent.mkdir(parents=True, exist_ok=True)

def load_indicateurs():
    if resultat_path.exists():
        with open(resultat_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"texte_complet": {"texte": "", "indicateurs": [], "phrases": []}}

def save_indicateurs(data):
    with open(resultat_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

txt = Text(texte="", phrase=[], indicateurs=[])

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/init")
def initialize_time():
    global time, started
    time = datetime.datetime.now() 
    started = 0  
    return {"message": "Temps initialisé avec succès", "start_time": time.isoformat()}


@app.post("/indic/sep")
def send_phrase(item: Text): 
    data = load_indicateurs()
    time2 = datetime.datetime.now()
    timeDiff = int((time2 - time).total_seconds())
    txt.texte += " " + item.texte 
    txt.phrase.append(item.texte) 
    
    indicateurs_phrase = morpho.stats_morpho_all(txt.texte, "indicateurs", timeDiff)

    txt.indicateurs = [indicateurs_phrase] 

    save_indicateurs(data)

    data["texte_complet"] = {
        "texte": txt.texte,
        "indicateurs": txt.indicateurs,  
        "phrases": txt.phrase
    }

    save_indicateurs(data)

    return {"texte": txt.texte, "indicateurs": txt.indicateurs, "texte_complet": data["texte_complet"]}

@app.get("/indic/")
def get_indicateurs_complets():
    data = load_indicateurs()
    return data.get("texte_complet", {})

@app.post("/indic_part")
def send_phrase_partiel(texte: dict): 
    print(texte)
    texte = texte["texte"]
    time2 = datetime.datetime.now()
    timeDiff = int((time2 - time).total_seconds())
    if(timeDiff == 0):
        timeDiff = 1
    if not texte.strip():  
        raise HTTPException(status_code=422, detail="Texte vide ou invalide.")
    indicateurs_phrase = morpho.stats_morpho_all(texte, "indicateurs", timeDiff)
    if not isinstance(indicateurs_phrase, dict):  
        raise HTTPException(status_code=500, detail="Format de données incorrect.")
    
    return {"indicateurs": indicateurs_phrase}

@app.get("/indic_phrase/{indice}")
def get_indicateurs_par_indice(indice: int):
    data = load_indicateurs()
    if "texte_complet" not in data or "phrases" not in data["texte_complet"]:
        raise HTTPException(status_code=404, detail="Aucune phrase enregistrée.")
    
    phrases = data["texte_complet"]["phrases"]
    indicateurs = data["texte_complet"]["indicateurs"]
    
    if indice < 0 or indice >= len(phrases):
        raise HTTPException(status_code=404, detail="Indice de phrase invalide.")
    
    phrase = phrases[indice]
    phrase_indicateurs = indicateurs[indice]
    
    return {"phrase": phrase, "indicateurs": phrase_indicateurs}

# pour tester, excuter uvicorn main:app --reload dans l'environnement
# requete type : 
# type : POST
# URL : http://127.0.0.1:8000/indic/
# body (raw, JSON) : {  "texte": "Aujourd'hui il y'as beaucoup de nuages"}
# modification encore nécessaire pour le fichier indicateurs.json
# potentiellement retiré la gestion des indicateurs mis en place pour faire de la simple lecture dans le fichier result_indicateurs.json