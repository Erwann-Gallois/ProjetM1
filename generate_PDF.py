import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
from xhtml2pdf import pisa
import base64

# Fonction pour convertir une figure matplotlib en image base64
def fig_to_base64(fig):
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')

# Génération d'un graphique en barres
def generate_bar_chart(title, categories, values):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(categories, values, color='blue')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel("Fréquence")
    ax.set_title(title)
    return fig_to_base64(fig)

# Génération d'un graphique en camembert
def generate_pie_chart(title, categories, values):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Assure un affichage circulaire
    ax.set_title(title)
    return fig_to_base64(fig)

# Génération d'un nuage de mots
def generate_wordcloud(word_frequencies):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig_to_base64(fig)

# Convertit un fichier HTML en PDF
def generate_pdf_from_html(html_content, pdf_filename):
    with open(pdf_filename, "wb") as result_file:
        pisa_status = pisa.CreatePDF(html_content, dest=result_file)
    return pdf_filename if not pisa_status.err else None

# Convertit une image locale en base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Fonction principale pour générer le rapport PDF
def generate_pdf_report(json_file, pdf_filename="rapport.pdf"):
    # Charger les données JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Génération des images en base64
    wordcloud_img = generate_wordcloud(data['has_unit'])

    # Définition des émotions et de leurs valeurs
    emotions = {
        "Joie": data["joy_rate"],
        "Peur": data["fear_rate"],
        "Tristesse": data["sadness_rate"],
        "Colère": data["anger_rate"],
        "Surprise": data["surprise_rate"],
        "Dégoût": data["disgust_rate"]
    }
    emotion_labels = list(emotions.keys())
    emotion_values = list(emotions.values())

    # Générer le graphique en camembert pour les émotions
    emotions_pie_img = generate_pie_chart("Répartition des émotions", emotion_labels, emotion_values)

    # Définition des métriques de concepts
    concept_metrics = {
        "Densité des concepts uniques": data["unique_concept_density"],
        "Efficacité des concepts uniques": data["unique_concept_efficiency"],
        "Densité des concepts totaux": data["total_concept_density"],
        "Efficacité des concepts totaux": data["total_concept_efficiency"]
    }
    concept_labels = list(concept_metrics.keys())
    concept_values = list(concept_metrics.values())
    concept_chart_img = generate_bar_chart("Analyse des Concepts", concept_labels, concept_values)

    # Génération du résumé des émotions avec un affichage structuré
    emotion_rows = "".join(f"""
        <tr>
            <td style="padding: 5px; border-bottom: 1px dotted gray; text-align: left; vertical-align: middle;">{label}</td>
            <td style="padding: 5px; border-bottom: 1px dotted gray; text-align: right; vertical-align: middle;">{value:.3f}</td>
        </tr>
    """ for label, value in emotions.items())

    emotion_summary = f"""
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th style="border-bottom: 2px solid black; padding: 5px; text-align: left;">Émotion</th>
                    <th style="border-bottom: 2px solid black; padding: 5px; text-align: right;">Résultat</th>
                </tr>
            </thead>
            <tbody>
                {emotion_rows}
            </tbody>
        </table>
    """

    # Génération du résumé des concepts avec un affichage structuré
    concept_rows = "".join(f"""
        <tr>
            <td style="padding: 5px; border-bottom: 1px dotted gray; text-align: left; vertical-align: middle;">{label}</td>
            <td style="padding: 5px; border-bottom: 1px dotted gray; text-align: right; vertical-align: middle;">{value:.5f}</td>
        </tr>
    """ for label, value in concept_metrics.items())

    concept_summary = f"""
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th style="border-bottom: 2px solid black; padding: 5px; text-align: left;">Concept</th>
                    <th style="border-bottom: 2px solid black; padding: 5px; text-align: right;">Valeur</th>
                </tr>
            </thead>
            <tbody>
                {concept_rows}
            </tbody>
        </table>
    """


    # Charger le logo CHU
    logoCHU = image_to_base64("images/logoCHU.png")

    # Génération du contenu HTML
    html_content = f"""
    <html>
    <head>
    <style>
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        td {{
            vertical-align: middle;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: auto;
        }}
        .vide{{
            height: 50px; 
        }}
    </style>
    </head>
    <body>

    <table class="header-table" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td class="left-header" style="width: 50%; vertical-align: top;">
                <h2>Université de Caen Basse-Normandie</h2>
                <img src="data:image/png;base64,{logoCHU}" alt="Logo CHU" width="150"/>
                <p>Av de la Côte de Nacre - CS 30 001, 14 033 Caen Cedex 09<br>Tel : 02.31.06.31.06</p>
                <div style="height: 70px;">&nbsp;</div>
                <div style="height: 70px;">&nbsp;</div>

                <h1>Rapport d'Analyse</h1>
            </td>
            
            <td class="right-header" style="width: 50%; vertical-align: middle; padding: 20px; display: flex; align-items: center; justify-content: space-between;">
                
                <div style="text-align: left; flex-grow: 1; padding-left:100px;">
                    <div>&nbsp;</div>
                    <div>&nbsp;</div>
                    <h2 style="margin-top: 0;">Rapport Médical</h2>
                    <p><strong>POLE DE PSYCHIATRIE</strong><br>Laboratoire de Psychologie<br>Responsable : Jean-Pierre Lépine</p>
                </div>
                <img src="data:image/png;base64,{wordcloud_img}" width="350px" alt="Nuage de mots" style="margin-right: 20px;"/>

            </td>


        </tr>
    </table>

        <hr>
        <h1>Analyse des émotions:</h1>

        <table style="width: 100%; text-align: center; border-collapse: collapse;">
            <tr>
                <td style=" text-align: left; vertical-align: top;">
                    {emotion_summary}
                </td>
                <td style="padding: 5px;">
                    <img src="data:image/png;base64,{emotions_pie_img}" width="300px" alt="Graphique des émotions" />
                </td>
            </tr>
        </table>

        <div>&nbsp;</div>
        <div>&nbsp;</div>
        <hr>
        <div>&nbsp;</div>
        
        <h1>Analyse des Concepts:</h1>

        <table style="width: 100%; text-align: center; border-collapse: collapse;">
            <tr>
                <td style="text-align: left; vertical-align: top;">
                    {concept_summary}
                </td>
                <td style="padding: 5px;">
                    <img src="data:image/png;base64,{concept_chart_img}" width="400px" alt="Graphique des concepts" />
                </td>
            </tr>
        </table>

    </body>
    </html>
    """

    # Génération du fichier PDF
    pdf_generated = generate_pdf_from_html(html_content, "resultat/"+ pdf_filename)
    return pdf_filename if pdf_generated else "Erreur de génération du PDF"


#exemple d'utilisation:
#generate_pdf_report("resultat/result_indicateurs.json", "rapport.pdf")
