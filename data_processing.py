import os
import logging
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import docx
from striprtf.striprtf import rtf_to_text
import olefile
import random
import csv

def extract_categories(config):
    root_dir = config['Paths']['documents']
    categories = set()

    for root, dirs, files in os.walk(root_dir):
        # Überspringe leere Ordner
        if not files:
            continue

        rel_path = os.path.relpath(root, root_dir)
        path_parts = rel_path.split(os.sep)

        if path_parts[0] == "LRH":
            category = f"LRH-{'-'.join(path_parts[1:])}"
        elif path_parts[0] == "unknown":
            category = "unknown"
        else:
            category = f"Ghostwriter-{path_parts[0]}"

        categories.add(category)

    return sorted(list(categories))


def save_categories_to_csv(categories, config):
    categories_file = os.path.join(config['Paths']['output'], 'categories.csv')
    with open(categories_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for category in categories:
            writer.writerow([category])

    print(f"Kategorien wurden in {categories_file} gespeichert.")
    print(f"Anzahl der Kategorien: {len(categories)}")
    logging.info(f"Kategorien wurden in {categories_file} gespeichert.")
    logging.info(f"Anzahl der Kategorien: {len(categories)}")


def get_files_and_categories(config):
    root_dir = config['Paths']['documents']
    files_and_categories = []
    supported_formats = ['.txt', '.rtf', '.doc', '.docx']

    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(fmt) for fmt in supported_formats):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, root_dir)

                # Binäre Klassifikation
                if rel_path.startswith("LRH"):
                    category = "LRH"
                else:
                    category = "Nicht-LRH"

                files_and_categories.append((file_path, category))

    return files_and_categories


def load_categories_from_csv(config):
    categories_file = os.path.join(config['Paths']['output'], 'categories.csv')
    if not os.path.exists(categories_file):
        raise FileNotFoundError(f"Kategorien-Datei nicht gefunden: {categories_file}. Bitte führen Sie zuerst update_categories.py aus.")

    with open(categories_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [row[0] for row in reader]


def create_dataset(config, quick=False):
    files_and_categories = get_files_and_categories(config)

    if quick:
        sample_size = max(1, int(len(files_and_categories) * 0.1))
        files_and_categories = random.sample(files_and_categories, sample_size)

    texts = []
    labels = []
    filenames = []

    for file_path, category in files_and_categories:
        text = extract_text_from_file(file_path)
        if text:
            texts.append(text)
            labels.append(category)
            filenames.append(os.path.basename(file_path))

    if not texts:
        raise ValueError("Keine Textdaten gefunden. Überprüfen Sie das Dokumentenverzeichnis.")

    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)

    dataset_dict = {
        'text': texts,
        'labels': numeric_labels,
        'filename': filenames
    }

    return Dataset.from_dict(dataset_dict), le


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
