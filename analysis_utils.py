import logging
import os
import torch
import gc
import csv
import sys

from datetime import date
from model_utils import predict_top_n, predict_for_model
from datetime import datetime
from colorama import Fore, Style
from sklearn.preprocessing import LabelEncoder
import numpy as np


def analyze_documents_csv(check_folder, models, extract_text_from_file):
    files = [f for f in os.listdir(check_folder) if os.path.isfile(os.path.join(check_folder, f))]

    # CSV-Header ausgeben
    print("Doc-Name,Modell,LRH-%,non-LRH-%")

    for file in tqdm(files, desc="Analysiere Dateien"):
        file_path = os.path.join(check_folder, file)
        results = analyze_document(file_path, models, extract_text_from_file)

        if results:
            for model_name, predictions in results:
                lrh_prob = next((prob for cat, prob in predictions if cat == "LRH"), 0)
                non_lrh_prob = next((prob for cat, prob in predictions if cat == "Nicht-LRH"), 0)

                # CSV-Zeile ausgeben
                print(f"{file},{model_name},{lrh_prob*100:.1f},{non_lrh_prob*100:.1f}")
        else:
            # Wenn keine Ergebnisse, trotzdem eine Zeile ausgeben
            print(f"{file},Keine Ergebnisse,0.0,0.0")

    sys.stdout.flush()  # Stellen Sie sicher, dass alle Ausgaben geschrieben wurden
    return


def analyze_document(file_path, models, extract_text_from_file):
    text = extract_text_from_file(file_path)
    if text is None or len(text) == 0:
        print(f"\nKonnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    results = []
    for model_name, (model, tokenizer, le) in models.items():
        torch.cuda.empty_cache()
        gc.collect()

        try:
            model.to(model.device)
            predictions = predict_for_model(model, tokenizer, text, le)
            results.append((model_name, predictions))
        except Exception as e:
            print(f"\nFehler bei der Verarbeitung von {file_path} mit Modell {model_name}: {str(e)}")
        finally:
            model.to('cpu')

    return results


def analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file):
    text = extract_text_from_file(file_path)

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    file_size = len(text.encode('utf-8'))
    top_predictions = predict_top_n(trainer, tokenizer, text, le)

    print(f"Vorhersagen für {os.path.basename(file_path)}:")
    logging.info(f"Vorhersagen für {os.path.basename(file_path)}:")
    for category, prob in top_predictions:
        print(f"{category}: {prob*100:.1f}%")
        logging.info(f"{category}: {prob*100:.1f}%")

    lrh_probability = next((prob for cat, prob in top_predictions if cat == "LRH"), 0)
    nicht_lrh_probability = next((prob for cat, prob in top_predictions if cat == "Nicht-LRH"), 0)

    threshold = 0.1  # 10% Unterschied als Schwellenwert
    if abs(lrh_probability - nicht_lrh_probability) < threshold:
        conclusion = "Nicht eindeutig"
    elif lrh_probability > nicht_lrh_probability:
        conclusion = "Wahrscheinlich LRH"
    else:
        conclusion = "Wahrscheinlich Nicht-LRH"

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße": file_size,
        "Datum": datetime.now().strftime("%d-%m-%y %H:%M"),
        "LRH": f"{lrh_probability:.2f}",
        "Nicht-LRH": f"{nicht_lrh_probability:.2f}",
        "Schlussfolgerung": conclusion
    }
