import hashlib
import time
from datetime import datetime
import json
import os
import olefile
import csv
import logging
import configparser
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaTokenizer
import torch
from striprtf.striprtf import rtf_to_text
import docx
import pdfplumber
from analysis_utils import analyze_new_article


def calculate_documents_checksum(documents_path):
    checksum = hashlib.md5()
    for root, _, files in os.walk(documents_path):
        for file in sorted(files):  # Sortieren für Konsistenz
            checksum.update(file.encode())
    return checksum.hexdigest()



def update_config_checksum(config, new_checksum):
    config['DocumentsCheck']['checksum'] = new_checksum
    with open('config.txt', 'w') as configfile:
        config.write(configfile)



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

    required_sections = ['Paths', 'Models', 'Training', 'Optimization', 'Evaluation']
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Abschnitt '{section}' fehlt in der config.txt")

    required_keys = {
        'Paths': ['documents', 'check_this', 'output', 'models'],
        'Models': ['model_list'],
        'Training': ['batch_size', 'learning_rate', 'num_epochs', 'weight_decay', 'warmup_steps', 'gradient_accumulation_steps'],
        'Optimization': ['fp16', 'max_grad_norm'],
        'Evaluation': ['eval_steps', 'save_steps']
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

    model_list = config['Models']['model_list'].split('\n')

    base_model_path = os.path.join(os.path.dirname(__file__), 'fresh-models')
    logging.info(f"Base model path: {base_model_path}")

    for model_line in model_list:
        if not model_line.strip():
            continue
        model_info = model_line.split(',')
        if len(model_info) != 4:
            raise ValueError(f"Ungültiges Modell-Format in config.txt: {model_line}")
        model_name, short_name, _, _ = model_info
        model_name = model_name.strip()

        local_model_path = os.path.join(base_model_path, model_name)
        logging.info(f"Versuche Modell zu laden: {model_name} von Pfad: {local_model_path}")

        try:
            if model_name == "Meta-Llama-3-8B":
                if not os.path.exists(local_model_path):
                    raise ValueError(f"Llama-Modell nicht gefunden: {local_model_path}")
                logging.info(f"Llama-Modell gefunden: {local_model_path}")
                tokenizer = LlamaTokenizer.from_pretrained(local_model_path)
                model = LlamaForCausalLM.from_pretrained(local_model_path)
            else:
                if os.path.exists(local_model_path):
                    logging.info(f"Lokales Modell gefunden: {local_model_path}")
                    tokenizer_json_path = os.path.join(local_model_path, 'tokenizer.json')
                    if os.path.exists(tokenizer_json_path):
                        logging.info(f"Tokenizer JSON gefunden: {tokenizer_json_path}")
                        with open(tokenizer_json_path, 'r') as f:
                            tokenizer_config = json.load(f)
                        tokenizer = AutoTokenizer.from_pretrained(
                            local_model_path,
                            use_fast=False,
                            tokenizer_config=tokenizer_config
                        )
                    else:
                        logging.info(f"Tokenizer JSON nicht gefunden, verwende Standard-Konfiguration")
                        tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)
                    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
                else:
                    logging.info(f"Lokales Modell nicht gefunden, versuche von Hugging Face zu laden: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)

            logging.info(f"Modell erfolgreich geladen: {model_name}")
        except Exception as e:
            logging.error(f"Fehler beim Laden des Modells {model_name}: {str(e)}")
            raise ValueError(f"Ungültiges oder nicht verfügbares Modell: {model_name}. Fehler: {str(e)}")

    if torch.cuda.is_available():
        logging.info(f"GPU verfügbar: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("Keine GPU verfügbar. Das Training wird auf der CPU durchgeführt und kann sehr lange dauern.")

    return config



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


def check_files(trainer, tokenizer, le, config, model_name):
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
        result = analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file)
        if result:
            result['Model'] = model_name
            results.append(result)

    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"Gesamtausführungszeit für Modell {model_name}: {total_duration:.2f} Sekunden")

    if results:
        csv_filename = os.path.join(config['Paths']['output'], "CheckThisResults.csv")
        file_exists = os.path.exists(csv_filename)

        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for result in results:
                writer.writerow(result)

        logging.info(f"Ergebnisse wurden an {csv_filename} angehängt.")
    else:
        logging.info("Keine Ergebnisse zur Ausgabe.")

    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"Gesamtausführungszeit für Modell {model_name}: {total_duration:.2f} Sekunden")

    if results:
        csv_filename = os.path.join(config['Paths']['output'], "CheckThisResults.csv")
        file_exists = os.path.exists(csv_filename)

        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for result in results:
                writer.writerow(result)

        logging.info(f"Ergebnisse wurden in {csv_filename} gespeichert.")

        lrh_count = sum(1 for r in results if r["Schlussfolgerung"] == "Wahrscheinlich LRH")
        logging.info(f"\nZusammenfassung: {len(results)} Dateien analysiert, {lrh_count} wahrscheinlich von LRH.")
    else:
        logging.info("Keine Ergebnisse zur Ausgabe.")


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

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
