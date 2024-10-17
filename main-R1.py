import time
from datetime import datetime
import os
import olefile
import csv
import logging
import configparser
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from striprtf.striprtf import rtf_to_text
import docx
import pdfplumber
from analysis_utils import analyze_new_article
from file_utils import check_environment, extract_text_from_file, get_device
from data_processing import load_categories_from_csv
from sklearn.preprocessing import LabelEncoder

def load_label_encoder(config):
    categories = load_categories_from_csv(config)
    if len(categories) != 2 or 'LRH' not in categories or 'Nicht-LRH' not in categories:
        raise ValueError("Die categories.csv Datei muss genau zwei Zeilen enthalten: 'Nicht-LRH' und 'LRH'")
    le = LabelEncoder()
    le.fit(categories)
    print("Label-Encoder Klassen:", le.classes_)
    return le


def check_hf_token():
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token is None:
        raise ValueError("Hugging Face API token is missing. Please set the environment variable 'HUGGINGFACE_HUB_TOKEN'.")
    api = HfApi()
    try:
        api.whoami(token=hf_token)
        logging.info("Hugging Face API token is valid.")
    except HTTPError as e:
        raise ValueError("Invalid Hugging Face API token. Please check the token and try again.")

def check_files(model, tokenizer, le, config, model_name):
    start_time = time.time()
    check_folder = config['Paths']['check_this']
    if not os.path.exists(check_folder):
        logging.error(f"Der Ordner '{check_folder}' existiert nicht.")
        return

    files = [f for f in os.listdir(check_folder) if os.path.isfile(os.path.join(check_folder, f))]

    if not files:
        logging.info(f"Keine Dateien im Ordner '{check_folder}' gefunden.")
        return

    results = []
    for file in files:
        file_path = os.path.join(check_folder, file)
        logging.info(f"Analysiere Datei: {file}")
        result = analyze_new_article(file_path, model, tokenizer, le, extract_text_from_file)
        if result:
            result['Model'] = model_name
            results.append(result)

    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"Gesamtausführungszeit für Modell {model_name}: {total_duration:.2f} Sekunden")

    if results:
        csv_filename = os.path.join(config['Paths']['output'], f"CheckThisResults_{model_name}.csv")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        logging.info(f"Ergebnisse wurden in {csv_filename} gespeichert.")

        lrh_count = sum(1 for r in results if r["Schlussfolgerung"] == "Wahrscheinlich LRH")
        logging.info(f"\nZusammenfassung für {model_name}: {len(results)} Dateien analysiert, {lrh_count} wahrscheinlich von LRH.")
    else:
        logging.info(f"Keine Ergebnisse zur Ausgabe für Modell {model_name}.")

def main():
    parser = argparse.ArgumentParser(description="Analyse von Texten mit verschiedenen Modellen")
    parser.add_argument("thema", help="Thema oder Bezeichnung für diesen Analysedurchlauf")
    parser.add_argument("--checkthis", action="store_true", help="Führe die Analyse für den 'check_this' Ordner durch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        check_hf_token()
        config, models, device = check_environment()

        le = load_label_encoder(config)

        timestamp = datetime.now().strftime("%y%m%d-%Hh%M")

        for model_name, (tokenizer, model) in models.items():
            model.to(device)
            logging.info(f"Verarbeite Modell: {model_name}")

            if args.checkthis:
                output_filename = f"{timestamp}-Results_{args.thema}_{model_name}.csv"
                output_path = os.path.join(config['Paths']['output'], output_filename)
                check_files(model, tokenizer, le, config, model_name, output_path)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        logging.info("Alle Modelle wurden verarbeitet.")

    except Exception as e:
        logging.error(f"Ein kritischer Fehler ist aufgetreten: {str(e)}")
        raise

if __name__ == "__main__":
    main()
