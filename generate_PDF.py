from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image, PageBreak
from datetime import datetime
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import base64
from reportlab.lib.units import inch

styles = getSampleStyleSheet()

# -----------------------------
# 📌 UTILITAIRES
# -----------------------------

def load_json(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Erreur : Impossible de charger {json_file}")
        return None

def fig_to_base64(fig):
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')

def generate_pie_chart(title, categories, values):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title(title)
    plt.axis('equal')
    return fig_to_base64(fig)
def generate_visual_summary_section(data):
    emotions = ["Colère", "Dégoût", "Peur", "Joie", "Tristesse", "Surprise"]
    emotion_keys = ["anger_rate", "disgust_rate", "fear_rate", "joy_rate", "sadness_rate", "surprise_rate"]
    emotion_values = [data.get(k, 0) for k in emotion_keys]
    emotion_chart = generate_pie_chart("Émotions détectées", emotions, emotion_values)
    emotion_img = Image(BytesIO(base64.b64decode(emotion_chart)), width=220, height=180)

    categories = ["Adj.", "Adv.", "Prep.", "Det.", "Nom", "Pron.", "Verbe", "Aux."]
    keys = ["adj_rate", "adv_rates", "adp_rate", "det_rate", "noun_rate", "pron_rate", "verb_rate", "verb_aux_rate"]
    syntax_labels = categories
    syntax_values = [data.get(k, 0) for k in keys]
    syntax_chart = generate_bar_chart("Répartition grammaticale", syntax_labels, syntax_values)
    syntax_img = Image(BytesIO(base64.b64decode(syntax_chart)), width=220, height=180)

    wordcloud_base64 = generate_wordcloud(data["has_unit"])
    wordcloud_image = Image(BytesIO(base64.b64decode(wordcloud_base64)), width=320, height=160)

    layout = Table([[emotion_img, syntax_img]], colWidths=[250, 250])
    layout.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))

    return [
        Paragraph("Résumé Visuel du Discours", styles["Heading2"]),
        Spacer(1, 6),
        layout,
        Spacer(1, 10),
        Paragraph("Nuage de mots", styles["Heading3"]),
        Spacer(1, 4),
        wordcloud_image,
   
    ]

def generate_bar_chart(title, categories, values):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, values, color='blue')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_title(title)
    return fig_to_base64(fig)

def generate_wordcloud(word_frequencies):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig_to_base64(fig)

def add_header(elements):
    university_text = Paragraph("<b><font size=14>Université de Caen Basse-Normandie</font></b>", styles["Normal"])
    header_text = Paragraph("<b>POLE DE PSYCHIATRIE</b><br/><b>Laboratoire de Psychologie</b>", styles["Normal"])
    location_text = Paragraph("""
    Av de la Côte de Nacre - CS 30 001<br/>
    14 033 Caen Cedex 09<br/>
    <b>Tel :</b> 02.31.06.31.06
    """, styles["Normal"])

    logo_path = "../images/logoCHU.png"
    try:
        logo_img = Image(logo_path, width=160, height=100)
    except:
        print("⚠️ Erreur : Impossible de charger le logo.")
        logo_img = None

    text_block = Table([
        [university_text],
        [header_text],
        [location_text]
    ], colWidths=[350])

    header_table = Table([
        [text_block, logo_img if logo_img else ""]
    ], colWidths=[350, 160])

    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 40))

def generate_wordcloud_section(data, width=360, height=180):
    """Génère une section contenant un WordCloud basé sur les mots utilisés."""
    wordcloud_img = generate_wordcloud(data['has_unit'])
    wordcloud_image = Image(BytesIO(base64.b64decode(wordcloud_img)), width=width, height=height)
    return [Spacer(1, 12), wordcloud_image, Spacer(1, 10)]


def generate_linguistic_analysis_section(data):
    elements = [Paragraph("Analyse Linguistique", styles["Heading2"]), Spacer(1, 4)]

    elements += generate_linguistic_stats_section(data)

    elements.append(Paragraph("Répartition grammaticale du discours", styles["Heading3"]))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        '<font size="8"><i>Ce tableau décrit la répartition des types de mots dans le discours, comme les verbes, noms ou adjectifs.</i></font>',
        styles["BodyText"]
    ))
    elements.append(Spacer(1, 4))

    raw_data = [
        ["Adjectifs", f"{data['adj_rate'] * 100:.1f}%"],
        ["Adverbes", f"{data['adv_rates'] * 100:.1f}%"],
        ["Prépositions", f"{data['adp_rate'] * 100:.1f}%"],
        ["Déterminants", f"{data['det_rate'] * 100:.1f}%"],
        ["Noms", f"{data['noun_rate'] * 100:.1f}%"],
        ["Pronoms", f"{data['pron_rate'] * 100:.1f}%"],
        ["Verbes", f"{data['verb_rate'] * 100:.1f}%"],
        ["Auxiliaires", f"{data['verb_aux_rate'] * 100:.1f}%"],
        ["Propositions avec sujet", f"{data['sub_prop_rate'] * 100:.1f}%"],
        ["Verbes avec sujet", f"{data['verb_subj_rate'] * 100:.1f}%"],
        ["Verbes avec objet", f"{data['verb_obj_rate'] * 100:.1f}%"],
        ["Verbes conjugués", f"{data['verb_conj_rate'] * 100:.1f}%"],
        ["Infinitifs", f"{data['verb_inf_rate'] * 100:.1f}%"],
        ["Taux de répétitions", f"{data['repetition_cons_rate']}"]
    ]

    table = Table([["Indicateur", "Valeur"]] + raw_data, colWidths=[300, 80])
    style = [
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 7),  
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.grey),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 0.5), 
    ('TOPPADDING', (0, 0), (-1, -1), 0.5)      
    ]
    for i in range(1, len(raw_data) + 1):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), colors.whitesmoke))
    table.setStyle(TableStyle(style))

    elements.append(table)
    elements.append(Spacer(1, 10))

    return elements

def generate_linguistic_stats_section(data):
    """Affiche les statistiques lexicales (TTR, Brunet, etc.) avec un texte explicatif et un style noir et blanc."""
    
    intro = Paragraph(
    '<font size="8"><i>Ce tableau présente des indicateurs de diversité et de richesse du vocabulaire utilisé. '
    'Il permet d’avoir un aperçu de la variété lexicale dans le discours du patient.</i></font>',
    styles["BodyText"]
)

    stats_data = [
        ["Indicateur", "Valeur"],
        ["Nombre total de mots", data["total_words"]],
        ["TTR (diversité lexicale)", f"{data['TTR']:.4f}"],
        ["Index de Brunet", f"{data['Brunet_index']:.2f}"],
        ["Statistique d'Honoré", f"{data['Honore_statistic']:.2f}"],
        ["Fréquence moyenne des mots", f"{data['mean_freq_words']:.2f}"],
        ["Écart-type de fréquence", f"{data['std_freq_words']:.2f}"],
        ["Plage de fréquence", data["range_freq_words"]]
    ]

    table = Table(stats_data, colWidths=[300, 80])
    style = [
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 7),  
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.grey),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 0.5),  
    ('TOPPADDING', (0, 0), (-1, -1), 0.5)      
    ]

    for i in range(1, len(stats_data)):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), colors.whitesmoke))

    table.setStyle(TableStyle(style))

    return [
        intro,
        Spacer(1, 6),
        Paragraph("Statistiques de diversité lexicale", styles["Heading3"]),
        Spacer(1, 4),
        table,
        Spacer(1, 10)
    ]

def table_to_image(data, filename="table_image.png"):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=["Indicateur", "Valeur"], loc="center", cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0, 1])
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return filename

def generate_sentiment_analysis_section(data):
    score = data.get("Score_AFINN", 0)
    interpretation = ""
    if score > 0:
        interpretation = "positif"
    elif score < 0:
        interpretation = "négatif"
    else:
        interpretation = "neutre"

    paragraph = Paragraph(
        f"<b>Analyse du sentiment général</b>",
        styles["Heading3"]
    )

    explanation = Paragraph(
        f'<font size="8"><i>Le score AFINN quantifie la tonalité émotionnelle globale du discours. '
        f'Ici, le score est de <b>{score}</b>, ce qui suggère un ton globalement <b>{interpretation}</b>.</i></font>',
        styles["BodyText"]
    )

    legend = Paragraph(
        '<font size="6"><i>Score > 0 : positif | ≈ 0 : neutre | < 0 : négatif</i></font>',
        styles["BodyText"]
    )

    return [paragraph, Spacer(1, 4), explanation, Spacer(1, 2), legend, Spacer(1, 10)]

def generate_emotion_table_section(data):
    """Affiche une table résumant les taux d’émotions détectées dans le discours."""
    intro = Paragraph(
        '<font size="8"><i>Ce tableau quantifie les émotions exprimées dans le discours.</i></font>',
        styles["BodyText"]
    )

    emotion_data = [
        ["Indicateur", "Valeur"],
        ["Colère (anger)", f"{data['anger_rate'] * 100:.2f} %"],
        ["Dégoût (disgust)", f"{data['disgust_rate'] * 100:.2f} %"],
        ["Peur (fear)", f"{data['fear_rate'] * 100:.2f} %"],
        ["Joie (joy)", f"{data['joy_rate'] * 100:.2f} %"],
        ["Tristesse (sadness)", f"{data['sadness_rate'] * 100:.2f} %"],
        ["Surprise", f"{data['surprise_rate'] * 100:.2f} %"]
    ]

    table = Table(emotion_data, colWidths=[300, 80])
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0.5),
        ('TOPPADDING', (0, 0), (-1, -1), 0.5)
    ]

    for i in range(1, len(emotion_data)):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), colors.whitesmoke))

    table.setStyle(TableStyle(style))

    return [
        Paragraph("Analyse Émotionnelle", styles["Heading3"]),
        Spacer(1, 4),
        intro,
        Spacer(1, 4),
        table,
        Spacer(1, 10)
    ]

# -----------------------------
# 📌 PDF MÉDECIN 
# -----------------------------

def generate_doctor_report(data, filename):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    add_header(elements)
    elements.append(Paragraph("<b><font size=16>Analyse Médicale</font></b>", styles["Title"]))
    elements.append(Spacer(1, 10))

    elements += generate_linguistic_analysis_section(data)
    elements += generate_emotion_table_section(data)
    elements += generate_sentiment_analysis_section(data)
    elements += generate_visual_summary_section(data)

    doc.build(elements)
    print(f"✅ Rapport médecin généré : {filename}")

# -----------------------------
# 📌 PDF PATIENT 
# -----------------------------

def generate_patient_title_section(data):
    summary = Paragraph("<b>Analyse de votre Consultation :</b>", styles["Heading2"])
    word_count = Paragraph(f"<b>Nombre de mots utilisés pendant l'échange :</b> {data['total_words']}", styles["Normal"])
    layout = Table([[summary, word_count]], colWidths=[350, 150])
    layout.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT')
    ]))
    return [layout, Spacer(1, 10)]

def generate_patient_wordcloud_section(data):
    wordcloud_img = generate_wordcloud(data['has_unit'])
    image = Image(BytesIO(base64.b64decode(wordcloud_img)), width=350, height=200)
    note = Paragraph(
        '<font size="8"><i>*Un nuage de mots est une représentation visuelle des mots les plus fréquents dans votre discours. '
        'Plus un mot est gros, plus il a été utilisé.</i></font>', styles["BodyText"])
    return [image, Spacer(1, 5), note, Spacer(1, 10)]

def generate_patient_emotion_syntax_section(data):
    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotion_values = [data[f"{emo}_rate"] for emo in emotions]
    filtered_emotions = [(emo, val) for emo, val in zip(emotions, emotion_values) if val > 0]
    emotion_img = None
    if filtered_emotions:
        emotion_labels, emotion_values = zip(*filtered_emotions)
        pie = generate_pie_chart("Répartition des émotions", emotion_labels, emotion_values)
        emotion_img = Image(BytesIO(base64.b64decode(pie)), width=230, height=200)

    syntax_categories = ["adj", "adp", "adv", "conj", "det", "noun", "pron", "verb", "propn"]
    syntax_values = [data.get(f"{cat}_rate", 0) for cat in syntax_categories]
    filtered_syntax = [(cat, val) for cat, val in zip(syntax_categories, syntax_values) if val > 0]
    syntax_img = None
    if filtered_syntax:
        syntax_labels, syntax_values = zip(*filtered_syntax)
        pie = generate_pie_chart("Analyse syntaxique", syntax_labels, syntax_values)
        syntax_img = Image(BytesIO(base64.b64decode(pie)), width=200, height=200)

    if emotion_img and syntax_img:
        layout = Table([[emotion_img, syntax_img]], colWidths=[250, 250])
        return [Paragraph("Vos émotions et votre analyse syntaxique:", styles["Heading2"]), layout, Spacer(1, 12)]
    return []

def generate_patient_report(data, filename):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    add_header(elements)
    elements += generate_patient_title_section(data)
    elements += generate_patient_wordcloud_section(data)
    elements += generate_patient_emotion_syntax_section(data)
    doc.build(elements)
    print(f"✅ Rapport patient généré : {filename}")

def generate_reports(json_file, pdf_basename="rapport", doctor_report=True, patient_report=True):
    data = load_json(json_file)
    if not data:
        print("Erreur: Impossible de charger les données")
        return
    if doctor_report:
        generate_doctor_report(data, f"rapport_medecin_{pdf_basename}.pdf")
    if patient_report:
        generate_patient_report(data, f"rapport_patient_{pdf_basename}.pdf")
    print("Rapports générés avec succès !")

# Exécution
if __name__ == "__main__":
    generate_reports("resultat/result_indicateurs.json", pdf_basename="analyse_final", doctor_report=True, patient_report=True)
