import pandas as pd
import os
import psycopg2
import json
import requests
import morpho

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
