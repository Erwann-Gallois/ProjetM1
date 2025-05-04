import pandas as pd
import psycopg2
import numpy as np


# 📌 Connexion PostgreSQL
connexion = psycopg2.connect(
    dbname="textes_db",
    user="utilisateur",
    password="mdp",
    host="localhost",
    port="5432"
)
curseur = connexion.cursor()

# Indicateur 
def get_indicateur(partie_de) : 
    """
    Permet de retourner les indicateurs de la base de données
    partie_de : int
    1 : Test
    2 : Validation
    3 : Train
    """
    if type(partie_de) != int : 
        raise ValueError("La partie doit être un entier")
    if partie_de > 3 or partie_de < 1 : 
        raise ValueError("La partie doit être comprise entre 1 et 3")
    
    sql = """
    SELECT indicateur FROM participants WHERE partof = %s
    """
    curseur.execute(sql, (partie_de,))
    resultats = curseur.fetchall()

    # Extraire les valeurs des tuples et les mettre dans une liste
    indicateurs = [row[0] for row in resultats]
    tableau_indicateurs = np.array(indicateurs)
    return tableau_indicateurs

def get_dialoguefr (partie_de) :
    """
    Permet de retourner les indicateurs de la base de données
    partie_de : int
    1 : Test
    2 : Validation
    3 : Train
    """
    if type(partie_de) != int : 
        raise ValueError("La partie doit être un entier")
    if partie_de > 3 or partie_de < 1 : 
        raise ValueError("La partie doit être comprise entre 1 et 3")
    
    sql = """
    SELECT dialogue_fr FROM participants WHERE partof = %s
    """
    curseur.execute(sql, (partie_de,))
    resultats = curseur.fetchall()

    # Extraire les valeurs des tuples et les mettre dans une liste
    dialogues = [row[0] for row in resultats]
    return dialogues

def get_labels(partie_de) :
    """
    Permet de retourner les labels de la base de données
    partie_de : int
    1 : Test
    2 : Validation
    3 : Train
    """
    if type(partie_de) != int : 
        raise ValueError("La partie doit être un entier")
    if partie_de > 3 or partie_de < 1 : 
        raise ValueError("La partie doit être comprise entre 1 et 3")
    
    sql = """
    SELECT PHQ_Binary FROM participants WHERE partof = %s
    """
    curseur.execute(sql, (partie_de,))
    resultats = curseur.fetchall()

    # Extraire les valeurs des tuples et les mettre dans une liste
    labels = [row[0] for row in resultats]
    return np.array(labels)