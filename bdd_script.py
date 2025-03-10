import pandas as pd
import os
import psycopg2
import json
import requests
from google import genai

# 📌 Définition de la clé API pour Gemini
client = genai.Client(api_key="AIzaSyC00_Cb218Ptx9WrI1NuOsiaEXNuY8RmMY")
# def translate_text(texts):
texts = ["Hello, how are you?", "I am fine, thank you.", "What are you doing?"]
response = client.models.generate_content(
    model = "gemini-2.0-flash",
    contents = "Translate the following sentences from English to French. Give only the traducted text and one option of traduction. Here are the sentences :" + "\n".join(texts),
)
try:
    print("🔁 Traduction en cours...")
    translated_text = response.text
    print(translated_text)
except (KeyError, IndexError):
    print("⚠️ Erreur lors de la traduction.")


