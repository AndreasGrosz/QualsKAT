import logging
import os
from datetime import date
from model_utils import predict_top_n
from datetime import datetime
from colorama import Fore, Style


def analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file):
    print(f"\nAnalysiere Datei: {os.path.basename(file_path)}")
    logging.info(f"Beginne Analyse von: {file_path}")

    text = extract_text_from_file(file_path)

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    logging.info(f"Extrahierter Text aus {file_path} (erste 50 Zeichen): {text[:50]}...")

    try:
        top_predictions, lrh_probability, ghostwriter_probability = predict_top_n(trainer, tokenizer, text, le, n=len(le.classes_))
    except Exception as e:
        logging.error(f"Fehler bei der Vorhersage für {file_path}: {str(e)}")
        return None

    file_size = len(text.encode('utf-8'))

    print("")
    print(f"{Fore.RED}Vorhersagen für {os.path.basename(file_path)}:{Style.RESET_ALL}")

    # Zeige nur die Top 5 Vorhersagen
    for category, prob in top_predictions[:5]:
        print(f"{category}: {prob:.4f}")

    print(f"LRH Gesamtwahrscheinlichkeit: {lrh_probability:.4f}")
    print(f"Ghostwriter Gesamtwahrscheinlichkeit: {ghostwriter_probability:.4f}")

    threshold = 0.1  # 10% Unterschied als Schwellenwert
    if abs(lrh_probability - ghostwriter_probability) < threshold:
        conclusion = "Nicht eindeutig"
    elif lrh_probability > ghostwriter_probability:
        conclusion = "Wahrscheinlich LRH"
    else:
        conclusion = "Wahrscheinlich Ghostwriter"

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße": file_size,
        "Datum": datetime.now().strftime("%d-%m-%y %H:%M"),
        "LRH": f"{lrh_probability:.4f}",
        "Ghostwriter": f"{ghostwriter_probability:.4f}",
        "Schlussfolgerung": conclusion
    }


