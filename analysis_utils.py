import logging
import os
from datetime import date
from model_utils import predict_top_n
from datetime import datetime
from colorama import Fore, Style
from sklearn.preprocessing import LabelEncoder
import numpy as np

def analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file):
    text = extract_text_from_file(file_path)

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    file_size = len(text.encode('utf-8'))

    # Erstelle einen neuen LabelEncoder und passe ihn an die Kategorien aus dem Modell an
    le = LabelEncoder()
    le.classes_ = np.array(list(trainer.model.config.id2label.values()))
    try:
        top_predictions, lrh_probability, ghostwriter_probability = predict_top_n(trainer, tokenizer, text, le, n=len(le.classes_))
    except KeyError as e:
        logging.error(f"Unbekannte Kategorie in der Vorhersage für {file_path}: {str(e)}")
        # Verwende eine Standardkategorie oder überspringe den Eintrag
        lrh_probability = "Nicht verfügbar"
        ghostwriter_probability = "Nicht verfügbar"
        top_predictions = []

    file_size = len(text.encode('utf-8'))

    print("")
    print(f"{Fore.RED}Vorhersagen für {os.path.basename(file_path)}:{Style.RESET_ALL}")
    logging.info("")
    logging.info(f"Vorhersagen für {os.path.basename(file_path)}:{Style.RESET_ALL}")

    # Zeige nur die Top 5 Vorhersagen
    for category, prob in top_predictions[:5]:
        print(f"{category}: {prob*100:.1f}%")
        logging.info(f"{category}: {prob*100:.1f}%")

    # Methode 1: Maximum-Wert
    lrh_max = max(prob for cat, prob in top_predictions if cat.startswith("LRH"))
    ghostwriter_max = max(prob for cat, prob in top_predictions if cat.startswith("Ghostwriter"))
    print(f"LRH Gesamtwahrscheinlichkeit (Maximum): {lrh_max*100:.1f}%")
    print(f"Ghostwriter Gesamtwahrscheinlichkeit (Maximum): {ghostwriter_max*100:.1f}%")
    logging.info(f"LRH Gesamtwahrscheinlichkeit (Maximum): {lrh_max*100:.1f}%")
    logging.info(f"Ghostwriter Gesamtwahrscheinlichkeit (Maximum): {ghostwriter_max*100:.1f}%")

    # Methode 2: Durchschnittswert
    lrh_mean = sum(prob for cat, prob in top_predictions if cat.startswith("LRH")) / sum(1 for cat, _ in top_predictions if cat.startswith("LRH"))
    ghostwriter_mean = sum(prob for cat, prob in top_predictions if cat.startswith("Ghostwriter")) / sum(1 for cat, _ in top_predictions if cat.startswith("Ghostwriter"))
    print(f"LRH Gesamtwahrscheinlichkeit (Durchschnitt): {lrh_mean*100:.1f}%")
    print(f"Ghostwriter Gesamtwahrscheinlichkeit (Durchschnitt): {ghostwriter_mean*100:.1f}%")
    logging.info(f"LRH Gesamtwahrscheinlichkeit (Durchschnitt): {lrh_mean*100:.1f}%")
    logging.info(f"Ghostwriter Gesamtwahrscheinlichkeit (Durchschnitt): {ghostwriter_mean*100:.1f}%")

    # Methode 3: Durchschnitt plus Standardabweichung
    lrh_std = (sum((prob - lrh_mean)**2 for cat, prob in top_predictions if cat.startswith("LRH")) / sum(1 for cat, _ in top_predictions if cat.startswith("LRH")))**0.5
    ghostwriter_std = (sum((prob - ghostwriter_mean)**2 for cat, prob in top_predictions if cat.startswith("Ghostwriter")) / sum(1 for cat, _ in top_predictions if cat.startswith("Ghostwriter")))**0.5
    lrh_probability = lrh_mean + 2 * lrh_std
    ghostwriter_probability = ghostwriter_mean + 2 * ghostwriter_std
    print(f"LRH Gesamtwahrscheinlichkeit (Durchschnitt + 2*Std): {lrh_probability*100:.1f}%")
    print(f"Ghostwriter Gesamtwahrscheinlichkeit (Durchschnitt + 2*Std): {ghostwriter_probability*100:.1f}%")
    logging.info(f"LRH Gesamtwahrscheinlichkeit (Durchschnitt + 2*Std): {lrh_probability*100:.1f}%")
    logging.info(f"Ghostwriter Gesamtwahrscheinlichkeit (Durchschnitt + 2*Std): {ghostwriter_probability*100:.1f}%")

    threshold = 0.1  # 10% Unterschied als Schwellenwert

    # Ermittle die Kategorie mit der höchsten Wahrscheinlichkeit
    if lrh_max > ghostwriter_max:
        max_category = "LRH"
    else:
        max_category = "Ghostwriter"

    print(f"Schlussfolgerung: {max_category}")
    logging.info(f"Schlussfolgerung: {max_category}")

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße": file_size,
        "Datum": datetime.now().strftime("%d-%m-%y %H:%M"),
        "LRH": f"{lrh_max*100:.1f}%",
        "Ghostwriter": f"{ghostwriter_max*100:.1f}%",
        "Schlussfolgerung": max_category
    }


