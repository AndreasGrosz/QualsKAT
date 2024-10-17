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


def save_results_to_csv(results, csv_filename):
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Dateiname', 'Dateigröße', 'Datum', 'LRH', 'Nicht-LRH', 'Schlussfolgerung', 'Model']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
            print(f"In CSV geschrieben: {result['Dateiname']} - LRH: {result['LRH']}, Nicht-LRH: {result['Nicht-LRH']}")


def analyze_documents_csv(check_folder, models, extract_text_from_file, output_file, save_interval=100):
    files = [f for f in os.listdir(check_folder) if os.path.isfile(os.path.join(check_folder, f))]

    header = ['Filename', 'r-base', 'ms-deberta', 'distilb', 'r-large', 'albert', 'Mittelwert']

    # Überprüfen, ob die Datei bereits existiert
    file_exists = os.path.isfile(output_file)

    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)

        for index, file in enumerate(tqdm(files, desc="Analysiere Dateien")):
            file_path = os.path.join(check_folder, file)
            text = extract_text_from_file(file_path)

            if text is None or len(text) == 0:
                output = [file, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
            else:
                results = []
                for model_name, (model, tokenizer, le) in models.items():
                    confidence = calculate_lrh_confidence(model, tokenizer, text, le)
                    results.append(confidence)

                avg_confidence = mean(results)
                output = [file] + [f"{conf:.1f}" for conf in results] + [f"{avg_confidence:.1f}"]

            writer.writerow(output)
            print(','.join(map(str, output)))

            # Zwischenspeichern in regelmäßigen Intervallen
            if (index + 1) % save_interval == 0:
                csvfile.flush()
                os.fsync(csvfile.fileno())
                print(f"Zwischenergebnis gespeichert nach {index + 1} Dateien.")

    print(f"Alle Ergebnisse wurden in {output_file} gespeichert.")

def calculate_lrh_confidence(model, tokenizer, text, le):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    print(f"Debug: probabilities = {probabilities}")
    lrh_index = le.transform(['LRH'])[0]
    print(f"Debug: LRH index according to LabelEncoder = {lrh_index}")
    lrh_probability = probabilities[lrh_index].item()
    print(f"Debug: LRH probability = {lrh_probability}")
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


def analyze_new_article(file_path, model, tokenizer, le, extract_text_func):
    logging.info(f"Beginne Analyse von: {os.path.basename(file_path)}")
    text = extract_text_func(file_path)
    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    top_predictions = predict_top_n(model, tokenizer, text, le)

    # Vereinfachte Zuordnung basierend auf dem Index
    lrh_probability = top_predictions[0][1]  # Annahme: Index 0 entspricht LRH
    nicht_lrh_probability = top_predictions[1][1]  # Annahme: Index 1 entspricht Nicht-LRH

    if abs(lrh_probability - nicht_lrh_probability) < 0.1:
        conclusion = "Nicht eindeutig"
    elif lrh_probability > nicht_lrh_probability:
        conclusion = "Wahrscheinlich LRH"
    else:
        conclusion = "Wahrscheinlich Nicht-LRH"

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße": len(text.encode('utf-8')),
        "Datum": datetime.now().strftime("%d-%m-%y %H:%M"),
        "LRH": f"{lrh_probability:.4f}",
        "Nicht-LRH": f"{nicht_lrh_probability:.4f}",
        "Schlussfolgerung": conclusion,
        "Model": model.name_or_path
    }

