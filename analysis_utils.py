import logging
import os
import torch
import gc
import csv
import sys
from tqdm import tqdm
from statistics import mean

from datetime import date
from model_utils import predict_top_n, predict_for_model
from datetime import datetime
from colorama import Fore, Style
from sklearn.preprocessing import LabelEncoder
import numpy as np


def analyze_documents_csv(check_folder, models, extract_text_from_file):
    files = [f for f in os.listdir(check_folder) if os.path.isfile(os.path.join(check_folder, f))]

    # CSV-Header
    header = ['Filename', 'r-base', 'ms-deberta', 'distilb', 'r-large', 'albert', 'Mittelwert']
    print(','.join(header))

    for file in files:
        file_path = os.path.join(check_folder, file)
        text = extract_text_from_file(file_path)

        if text is None or len(text) == 0:
            print(f'"{file}",N/A,N/A,N/A,N/A,N/A,N/A')
            continue

        results = []
        for model_name, (model, tokenizer, le) in models.items():
            confidence = calculate_lrh_confidence(model, tokenizer, text, le)
            results.append(confidence)

        avg_confidence = mean(results)

        # CSV-Zeile ausgeben
        output = [f'"{file}"'] + [f"{conf:.1f}" for conf in results] + [f"{avg_confidence:.1f}"]
        print(','.join(output))

    sys.stdout.flush()

def calculate_lrh_confidence(model, tokenizer, text, le):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    lrh_index = le.transform(['LRH'])[0]
    lrh_probability = probabilities[lrh_index].item()

    return lrh_probability * 100  # Umwandlung in Prozent



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
