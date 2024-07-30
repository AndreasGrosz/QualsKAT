import os
import time
import csv
import logging
import configparser
import argparse
from datetime import date
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='classifier.log')

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

def extract_text_from_file(file_path):
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_path.endswith('.rtf'):
            return handle_rtf_error(file_path)
        # Fügen Sie hier weitere Dateitypen hinzu, wenn nötig
        else:
            logging.error(f"Unsupported file type: {file_path}")
            return None
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return None

def handle_rtf_error(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            rtf_text = file.read()
            from striprtf.striprtf import rtf_to_text
            return rtf_to_text(rtf_text)
    except Exception as e:
        logging.error(f"Error reading .rtf file {file_path}: {str(e)}")
        return None

def get_files_and_categories(config):
    root_dir = config['Paths']['documents']
    files_and_categories = []
    supported_formats = ['.txt', '.rtf']  # Erweitern Sie diese Liste bei Bedarf

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

def get_model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def setup_model_and_trainer(dataset_dict, num_labels, config, model_name):
    tokenizer, model = get_model_and_tokenizer(model_name, num_labels)

    model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
    os.makedirs(model_save_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=float(config['Training']['learning_rate']),
        per_device_train_batch_size=int(config['Training']['batch_size']),
        per_device_eval_batch_size=int(config['Training']['batch_size']),
        num_train_epochs=int(config['Training']['num_epochs']),
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    if 'validation' not in dataset_dict:
        logging.warning("Kein Validierungsdatensatz gefunden. Verwende Testdatensatz für die Validierung.")
        eval_dataset = dataset_dict['test']
    else:
        eval_dataset = dataset_dict['validation']

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    return tokenizer, trainer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).astype(np.float32).mean().item()}

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

def analyze_new_article(file_path, trainer, tokenizer, le):
    text = extract_text_from_file(file_path)

    if text is None:
        return None

    file_size = len(text.encode('utf-8'))
    top_predictions = predict_top_n(trainer, tokenizer, text, le, n=5)

    lrh_probability = sum(conf for cat, conf in top_predictions if cat.startswith("LRH"))
    ghostwriter_probability = sum(conf for cat, conf in top_predictions if cat.startswith("Ghostwriter"))

    conclusion = "Wahrscheinlich LRH" if lrh_probability > ghostwriter_probability else "Wahrscheinlich nicht LRH"

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße (Bytes)": file_size,
        "Datum": date.today().strftime("%Y-%m-%d"),
        "LRH Wahrscheinlichkeit": f"{lrh_probability:.2f}",
        "Ghostwriter Wahrscheinlichkeit": f"{ghostwriter_probability:.2f}",
        "Schlussfolgerung": conclusion
    }

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
        result = analyze_new_article(file_path, trainer, tokenizer, le)
        if result:
            results.append(result)

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

def main():
    parser = argparse.ArgumentParser(description='LRH Document Classifier')
    parser.add_argument('--checkthis', action='store_true', help='Analyze files in the CheckThis folder')
    args = parser.parse_args()

    try:
        config = check_environment()
        check_hf_token()

        dataset, le = create_dataset(config)

        if len(dataset) == 0:
            raise ValueError("Der erstellte Datensatz ist leer. Überprüfen Sie die Eingabedaten.")

        train_testvalid = dataset.train_test_split(test_size=0.3)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        dataset_dict = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']
        })

        model_names = config['Model']['model_name'].split(',')
        total_start_time = time.time()

        for model_name in model_names:
            model_name = model_name.strip()
            logging.info(f"Training model: {model_name}")
            start_time = time.time()

            tokenizer, trainer = setup_model_and_trainer(dataset_dict, len(le.classes_), config, model_name)

            if not args.checkthis:
                trainer.train()
                results = trainer.evaluate(dataset_dict['test'])
                logging.info(f"Test results for {model_name}: {results}")

                # Save the model
                model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)

                end_time = time.time()
                logging.info(f"Training time for {model_name}: {(end_time - start_time) / 60:.2f} minutes")
            else:
                # Load the model for checking
                model_load_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
                model = AutoModelForSequenceClassification.from_pretrained(model_load_path)
                tokenizer = AutoTokenizer.from_pretrained(model_load_path)
                trainer = Trainer(model=model)
                check_files(trainer, tokenizer, le, config)

        total_end_time = time.time()
        total_duration = (total_end_time - total_start_time) / 60
        logging.info(f"Total execution time for all models: {total_duration:.2f} minutes")

    except Exception as e:
        logging.error(f"Ein kritischer Fehler ist aufgetreten: {str(e)}")
        raise

if __name__ == "__main__":
    main()
