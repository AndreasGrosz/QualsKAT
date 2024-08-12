import os
import time
from datetime import date
import logging
import configparser
import argparse
import sys
import json
from tqdm import tqdm
import gc
from colorama import Fore, Style
from datetime import date
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, set_seed
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

# Importe aus Ihren eigenen Modulen
from file_utils import check_environment, check_hf_token, check_files, extract_text_from_file
from data_processing import create_dataset, load_categories_from_csv
from model_utils import setup_model_and_trainer, get_model_and_tokenizer
from analysis_utils import analyze_new_article
from file_utils import get_device
from experiment_logger import log_experiment

# Aktualisiere den GradScaler
if hasattr(torch.cuda.amp, 'GradScaler'):
    torch.cuda.amp.GradScaler = lambda **kwargs: torch.amp.GradScaler('cuda', **kwargs)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='classifier.log')

def main():
    logging.info("")
    logging.error("Programmstart")
    logging.error("=============")

    # Setze einen festen Seed für Reproduzierbarkeit
    set_seed(42)

    device = get_device()
    logging.info(f"Verwende Gerät: {device}")

    # Aktiviere die Speichereffizienz von PyTorch
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description='LRH Document Classifier',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--train', action='store_true',
                        help='Trainiert das Modell mit den Dokumenten im "documents" Ordner.')
    parser.add_argument('--checkthis', action='store_true',
                        help='Analysiert Dateien im "CheckThis" Ordner mit einem trainierten Modell.')
    parser.add_argument('--quick', action='store_true',
                        help='Führt eine schnelle Analyse mit einem Bruchteil der Daten durch.')
    parser.add_argument('--predict', metavar='FILE',
                        help='Macht eine Vorhersage für eine einzelne Datei.')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        config = check_environment()
        check_hf_token()

        categories_file = os.path.join(config['Paths']['output'], 'categories.csv')
        if not os.path.exists(categories_file):
            print("Kategorien-Datei nicht gefunden. Bitte führen Sie zuerst update_categories.py aus.")
            logging.error("Kategorien-Datei nicht gefunden. Bitte führen Sie zuerst update_categories.py aus.")
            sys.exit(1)

        categories = load_categories_from_csv(config)
        if len(categories) == 0:
            raise ValueError("Keine Kategorien gefunden. Bitte überprüfen Sie die categories.csv Datei.")

        dataset, le = create_dataset(config, quick=args.quick)
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
            logging.info(f"Verarbeite Modell: {model_name}")

            model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))

            if args.train:
                num_labels = len(categories)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label={i: cat for i, cat in enumerate(categories)},
                    label2id={cat: i for i, cat in enumerate(categories)}
                )

                tokenizer, trainer, tokenized_datasets = setup_model_and_trainer(dataset_dict, le, config, model_name, model, quick=args.quick)
                trainer.train()
                results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                logging.info(f"Testergebnisse für {model_name}: {results}")

                # Logging des Experiments
                log_experiment(config, model_name, results, config['Paths']['output'])

                logging.info(f"Trainings- und Evaluationsergebnisse:")
                for key, value in results.items():
                    logging.info(f"{key}: {value}")
                logging.info(f"Gesamtausführungszeit für Modell {model_name}: {(time.time() - total_start_time) / 60:.2f} Minuten")
                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)

            if args.checkthis or args.predict:
                model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)

                # Laden der Kategorie-zu-Label-Zuordnung
                label2id = model.config.label2id
                id2label = model.config.id2label
                print("Modellkategorien:", id2label)
                logging.info("Modellkategorien: %s", id2label)

                # Erstellen eines neuen LabelEncoders mit den geladenen Kategorien
                le = LabelEncoder()
                le.classes_ = np.array(list(label2id.keys()))

                trainer = Trainer(model=model)

                if args.checkthis:
                    models = {}
                    for model_name in model_names:
                        model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
                        model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
                        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                        le = LabelEncoder()
                        le.classes_ = np.array(list(model.config.id2label.values()))
                        models[model_name] = (model, tokenizer, le)

                    check_folder = config['Paths']['check_this']
                    files = [f for f in os.listdir(check_folder) if os.path.isfile(os.path.join(check_folder, f))]

                    for file in tqdm(files, desc="Analysiere Dateien"):
                        file_path = os.path.join(check_folder, file)
                        results = analyze_document(file_path, models, extract_text_from_file)

                        if results:
                            print(f"\nVorhersagen für {file}:")
                            for model_name, predictions in results:
                                print(f"Modell: {model_name}")
                                for category, prob in sorted(predictions, key=lambda x: x[0]):
                                    print(f"{category}: {prob*100:.1f}%")
                                print()  # Leerzeile zwischen den Modellen

                if args.predict:
                    result = analyze_new_article(args.predict, trainer, tokenizer, le, extract_text_from_file)
                    if result:
                        print(json.dumps(result, indent=2))
                        logging.info(json.dumps(result, indent=2))
                    else:
                        print("Konnte keine Analyse durchführen.")
                        logging.info("Konnte keine Analyse durchführen.")

        total_end_time = time.time()
        total_duration = (total_end_time - total_start_time) / 60
        logging.error(f"Gesamtausführungszeit für alle Modelle: {total_duration:.2f} Minuten")
        logging.error("---------------------------------------")
        logging.error("Programmende")

    except Exception as e:
        logging.error(f"Ein kritischer Fehler ist aufgetreten: {str(e)}")
        raise

if __name__ == "__main__":
    main()
