import time
from datetime import datetime
import os
import olefile
import csv
import logging
import configparser
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
from striprtf.striprtf import rtf_to_text
import docx
import pdfplumber
from analysis_utils import analyze_new_article, save_results_to_csv
import re

def check_environment():
    device = get_device()
    if device.type == "cuda":
        logging.info(f"GPU verfügbar: {torch.cuda.get_device_name(0)}")
        logging.info(f"Verfügbarer GPU-Speicher: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logging.warning("Keine GPU verfügbar. Das Training wird auf der CPU durchgeführt und kann sehr lange dauern.")

    if not os.path.exists('config.txt'):
        raise FileNotFoundError("config.txt nicht gefunden. Bitte stellen Sie sicher, dass die Datei im aktuellen Verzeichnis liegt.")

    config = configparser.ConfigParser()
    config.read('config.txt')

    required_sections = ['Paths', 'Training', 'Model']
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Abschnitt '{section}' fehlt in der config.txt")

    required_keys = {
        'Paths': ['documents', 'check_this', 'output', 'models'],
        'Training': ['batch_size', 'learning_rate', 'num_epochs'],
        'Model': ['model_name']
    }
    for section, keys in required_keys.items():
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Schlüssel '{key}' fehlt im Abschnitt '{section}' der config.txt")

    for path_key in ['documents', 'check_this', 'output', 'models']:
        path = config['Paths'][path_key]
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                logging.info(f"Verzeichnis '{path}' wurde erstellt.")
            except Exception as e:
                raise OSError(f"Konnte Verzeichnis '{path}' nicht erstellen: {str(e)}")

    documents_path = config['Paths']['documents']
    if not any(os.scandir(documents_path)):
        raise FileNotFoundError(f"Keine Dateien im Verzeichnis '{documents_path}' gefunden.")

    models = {}
    model_names = config['Model']['model_name'].split(',')
    base_path = config['Paths']['models']

    for model_name in model_names:
        model_name = model_name.strip()
        if not model_name:
            logging.warning("Leerer Modellname in der Konfiguration gefunden. Bitte überprüfen Sie die config.txt")
            continue

        tokenizer, model = load_model(model_name, base_path)
        if tokenizer and model:
            models[model_name] = (tokenizer, model)
        else:
            logging.error(f"Modell {model_name} konnte nicht geladen werden.")

    if not models:
        raise ValueError("Keine Modelle konnten geladen werden. Bitte überprüfen Sie die Modellpfade und Namen.")

    return config, models, device

def load_model(model_name, base_path):
    try:
        logging.info(f"Versuche, Modell zu laden: {model_name}")
        model_path = os.path.join(base_path, model_name)

        if not os.path.exists(model_path):
            raise ValueError(f"Modellverzeichnis nicht gefunden: {model_path}")

        logging.debug(f"Inhalt des Modellverzeichnisses: {os.listdir(model_path)}")

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"config.json nicht gefunden in: {model_path}")

        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, local_files_only=True)

        logging.info(f"Modell erfolgreich geladen: {model_name}")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Fehler beim Laden des Modells {model_name}: {str(e)}", exc_info=True)
        return None, None

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


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


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def check_files(model, tokenizer, le, config, model_name):
    start_time = time.time()
    check_folder = config['Paths']['check_this']
    if not os.path.exists(check_folder):
        logging.error(f"Der Ordner '{check_folder}' existiert nicht.")
        return

    # Holen und sortieren Sie alle Dateien im Voraus
    all_files = [f for f in os.listdir(check_folder) if os.path.isfile(os.path.join(check_folder, f))]
    all_files.sort(key=natural_sort_key)

    logging.info("Sortierte Dateiliste:")
    for file in all_files:
        logging.info(file)

    if not all_files:
        logging.info(f"Keine Dateien im Ordner '{check_folder}' gefunden.")
        return

    results = []
    for file in all_files:
        file_path = os.path.join(check_folder, file)
        logging.info(f"Analysiere Datei: {file}")
        result = analyze_new_article(file_path, model, tokenizer, le, extract_text_from_file)
        if result:
            result['Model'] = model_name
            results.append(result)
            print(f"Verarbeitet: {result['Dateiname']} - LRH: {result['LRH']}, Nicht-LRH: {result['Nicht-LRH']}")

    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"Gesamtausführungszeit für Modell {model_name}: {total_duration:.2f} Sekunden")

    if results:
        csv_filename = os.path.join(config['Paths']['output'], f"CheckThisResults_{model_name}.csv")
        save_results_to_csv(results, csv_filename)
        logging.info(f"Ergebnisse wurden in {csv_filename} gespeichert.")

        lrh_count = sum(1 for r in results if r["Schlussfolgerung"] == "Wahrscheinlich LRH")
        logging.info(f"\nZusammenfassung für {model_name}: {len(results)} Dateien analysiert, {lrh_count} wahrscheinlich von LRH.")
    else:
        logging.info(f"Keine Ergebnisse zur Ausgabe für Modell {model_name}.")

def extract_text_from_file(file_path):

    if file_path.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logging.info(f"Erfolgreich gelesen: {file_path}")
            return text
        except Exception as e:
            logging.error(f"Fehler beim Lesen von {file_path}: {str(e)}")
            return None

    elif file_path.endswith('.rtf'):
        return handle_rtf_error(file_path)
    elif file_path.endswith('.pdf'):
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages])
            return text
        except Exception as e:
            logging.error(f"Fehler beim Lesen der PDF-Datei {file_path}: {str(e)}")
            return None
    elif file_path.endswith('.docx') or file_path.endswith('.doc'):
        try:
            doc = docx.Document(file_path)
            return ' '.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logging.warning(f"Could not open {file_path} as a .docx file. Attempting old .doc format extraction.")
            return extract_text_from_old_doc(file_path)
    else:
        logging.error(f"Unsupported file type: {file_path}")
        return None


def extract_text_from_old_doc(file_path):
    try:
        if olefile.isOleFile(file_path):
            with olefile.OleFileIO(file_path) as ole:
                streams = ole.listdir()
                if ole.exists('WordDocument'):
                    word_stream = ole.openstream('WordDocument')
                    text = word_stream.read().decode('utf-16le', errors='ignore')
                    return ' '.join(text.split())
                else:
                    logging.error(f"No WordDocument stream found in {file_path}")
                    return None
        else:
            logging.error(f"{file_path} is not a valid OLE file")
            return None
    except Exception as e:
        logging.error(f"Error reading old .doc file {file_path}: {str(e)}")
        return None


def handle_rtf_error(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            rtf_text = file.read()
            return rtf_to_text(rtf_text)
    except Exception as e:
        logging.error(f"Error reading .rtf file {file_path}: {str(e)}")
        return None


