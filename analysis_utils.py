import logging
import os
from datetime import date
from model_utils import predict_top_n
from datetime import datetime

def analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file):
    logging.info(f"Beginne Analyse von: {file_path}")

    text = extract_text_from_file(file_path)

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    logging.info(f"Extrahierter Text aus {file_path} (erste 50 Zeichen): {text[:50]}...")

    try:
        top_predictions = predict_top_n(trainer, tokenizer, text, le, n=5)
    except Exception as e:
        logging.error(f"Fehler bei der Vorhersage für {file_path}: {str(e)}")
        return None

    file_size = len(text.encode('utf-8'))
    logging.info(f"Dateigröße: {file_size} Bytes")

    lrh_probability = sum(conf for cat, conf in top_predictions if cat.startswith("LRH"))
    ghostwriter_probability = sum(conf for cat, conf in top_predictions if cat.startswith("Ghostwriter"))

    threshold = 0.1  # 10% Unterschied als Schwellenwert
    if abs(lrh_probability - ghostwriter_probability) < threshold:
        conclusion = "Nicht eindeutig"
    elif lrh_probability > ghostwriter_probability:
        conclusion = "Wahrscheinlich LRH"
    else:
        conclusion = "Wahrscheinlich Ghostwriter"

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße (Bytes)": file_size,
        "Datum": datetime.now().strftime("%d-%m-%y %H:%M"),
        "LRH Wahrscheinlichkeit": f"{lrh_probability:.2f}",
        "Ghostwriter Wahrscheinlichkeit": f"{ghostwriter_probability:.2f}",
        "Schlussfolgerung": conclusion
    }


