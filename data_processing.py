import os
import logging
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import docx
from striprtf.striprtf import rtf_to_text
import olefile

def get_files_and_categories(config):
    root_dir = config['Paths']['documents']
    files_and_categories = []
    supported_formats = ['.txt', '.rtf', '.doc', '.docx']

    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(fmt) for fmt in supported_formats):
                file_path = os.path.join(root, file)
                categories = os.path.relpath(root, root_dir).split(os.sep)
                if categories[0] == "LRH":
                    category = f"LRH-{'-'.join(categories[1:])}"
                elif categories[0] == "unknown":
                    category = "unknown"
                else:
                    category = f"Ghostwriter-{categories[0]}"
                files_and_categories.append((file_path, category))

    return files_and_categories

def create_dataset(config):
    files_and_categories = get_files_and_categories(config)
    texts = []
    all_categories = []

    for file_path, category in files_and_categories:
        text = extract_text_from_file(file_path)
        if text:
            texts.append(text)
            all_categories.append(category)

    if not texts:
        raise ValueError("Keine Textdaten gefunden. Überprüfen Sie das Dokumentenverzeichnis.")

    le = LabelEncoder()
    numeric_categories = le.fit_transform(all_categories)

    dataset_dict = {
        'text': texts,
        'labels': numeric_categories
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
