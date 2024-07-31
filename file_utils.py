# In file_utils.py
import os
from datetime import date
import olefile
import csv
import logging
import configparser
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
#import docx2txt
from striprtf.striprtf import rtf_to_text
import docx

# Funktionsdefinitionen hier...

def check_environment():
    # Überprüfe, ob die Konfigurationsdatei existiert
    if not os.path.exists('config.txt'):
        raise FileNotFoundError("config.txt nicht gefunden. Bitte stellen Sie sicher, dass die Datei im aktuellen Verzeichnis liegt.")

    # Lade die Konfiguration
    config = configparser.ConfigParser()
    config.read('config.txt')

    # Überprüfe, ob alle erforderlichen Abschnitte und Schlüssel in der Konfiguration vorhanden sind
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

    # Überprüfe, ob die angegebenen Pfade existieren
    for path_key in ['documents', 'check_this', 'output', 'models']:
        path = config['Paths'][path_key]
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                logging.info(f"Verzeichnis '{path}' wurde erstellt.")
            except Exception as e:
                raise OSError(f"Konnte Verzeichnis '{path}' nicht erstellen: {str(e)}")

    # Überprüfe, ob es Dokumente im documents-Verzeichnis gibt
    documents_path = config['Paths']['documents']
    if not any(os.scandir(documents_path)):
        raise FileNotFoundError(f"Keine Dateien im Verzeichnis '{documents_path}' gefunden.")

    # Überprüfe, ob die angegebenen Modelle gültig sind
    model_names = config['Model']['model_name'].split(',')
    for model_name in model_names:
        model_name = model_name.strip()
        try:
            AutoTokenizer.from_pretrained(model_name)
            AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Ungültiges oder nicht verfügbares Modell: {model_name}. Fehler: {str(e)}")

    # Überprüfe GPU-Verfügbarkeit
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

def check_files(trainer, tokenizer, le, config):
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
        result = analyze_new_article(file_path, trainer, tokenizer, le)
        if result:
            results.append(result)
            logging.info(f"Ergebnis für {file}: LRH: {result['LRH Wahrscheinlichkeit']}, "
                         f"Ghostwriter: {result['Ghostwriter Wahrscheinlichkeit']}, "
                         f"Schlussfolgerung: {result['Schlussfolgerung']}")
        else:
            logging.warning(f"Konnte keine Analyse für {file} durchführen.")

    csv_filename = os.path.join(config['Paths']['output'], "CheckThisResults.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    logging.info(f"Ergebnisse wurden in {csv_filename} gespeichert.")

    lrh_count = sum(1 for r in results if r["Schlussfolgerung"] == "Wahrscheinlich LRH")
    logging.info(f"\nZusammenfassung: {len(results)} Dateien analysiert, {lrh_count} wahrscheinlich von LRH.")


def extract_text_from_file(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252']

    if file_path.endswith('.txt'):
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        logging.error(f"Konnte {file_path} mit keinem der versuchten Encodings lesen.")
        return None
    elif file_path.endswith('.rtf'):
        return handle_rtf_error(file_path)
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
                    # Extrahiere Text aus dem WordDocument-Stream
                    word_stream = ole.openstream('WordDocument')
                    text = word_stream.read().decode('utf-16le', errors='ignore')
                    return ' '.join(text.split())  # Entferne überschüssige Leerzeichen
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



def extract_text_from_doc(content):
    text = []
    for i in range(len(content)):
        if content[i:i+1] == b'\x00' and content[i+1:i+2] != b'\x00':
            text.append(content[i+1:i+2].decode('utf-8', errors='ignore'))
    return ''.join(text)


def handle_rtf_error(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            rtf_text = file.read()
            return rtf_to_text(rtf_text)
    except Exception as e:
        logging.error(f"Error reading .rtf file {file_path}: {str(e)}")
        return None

def analyze_new_article(file_path, trainer, tokenizer, le):
    text = extract_text_from_file(file_path)

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    file_size = len(text.encode('utf-8'))
    top_predictions = predict_top_n(trainer, tokenizer, text, le, n=5)

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
        "Datum": date.today().strftime("%Y-%m-%d"),
        "LRH Wahrscheinlichkeit": f"{lrh_probability:.2f}",
        "Ghostwriter Wahrscheinlichkeit": f"{ghostwriter_probability:.2f}",
        "Schlussfolgerung": conclusion
    }

def predict_top_n(trainer, tokenizer, text, le, n=3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = trainer.model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_n_prob, top_n_indices = torch.topk(probabilities, n)
    results = []
    for prob, idx in zip(top_n_prob[0], top_n_indices[0]):
        category = le.inverse_transform([idx.item()])[0]
        results.append((category, prob.item()))
    return results
