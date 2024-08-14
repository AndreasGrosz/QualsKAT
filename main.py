import os
import time
from datetime import date
import logging
import configparser
import argparse
import hashlib
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
from huggingface_hub import HfApi, hf_hub_download
from requests.exceptions import HTTPError

# Importe aus Ihren eigenen Modulen
from file_utils import check_environment, check_hf_token, check_files, extract_text_from_file, calculate_documents_checksum, update_config_checksum
from data_processing import create_dataset, load_categories_from_csv
from model_utils import setup_model_and_trainer, get_model_and_tokenizer, get_models_for_task
from analysis_utils import analyze_new_article, analyze_document, analyze_documents_csv
from file_utils import get_device, extract_text_from_file
from experiment_logger import log_experiment


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

        categories = load_categories_from_csv(config)
        if len(categories) == 0:
            raise ValueError("Keine Kategorien gefunden. Bitte überprüfen Sie die categories.csv Datei.")

        dataset, le = create_dataset(config, quick=args.quick)
        if len(dataset) == 0:
            raise ValueError("Der erstellte Datensatz ist leer. Überprüfen Sie die Eingabedaten.")

        # models_to_process = get_models_for_task(config, 'train' if args.train else 'check')



        total_start_time = time.time()
        training_performed = False
        config = configparser.ConfigParser()
        config.read('config.txt')

        documents_path = config['Paths']['documents']
        current_checksum = calculate_documents_checksum(documents_path)
        stored_checksum = config['DocumentsCheck'].get('checksum', '')

        if args.train:
            training_performed = True
            models_to_process = get_models_for_task(config, 'train')

        if current_checksum != stored_checksum:
            logging.info("Änderungen im documents-Ordner erkannt. Starte vollständiges Neutraining.")
            for hf_name, short_name in models_to_process:
                logging.info(f"Trainiere Modell: {hf_name} ({short_name})")

                model_save_path = os.path.join(config['Paths']['models'], short_name)

                num_labels = len(categories)
                model, tokenizer = get_model_and_tokenizer(hf_name, num_labels, categories, config)

                trainer, tokenized_datasets = setup_model_and_trainer(dataset, le, config, hf_name, model, tokenizer, args.quick)

                try:
                    trainer.train()
                except ValueError as e:
                    if "does not support gradient checkpointing" in str(e):
                        logging.warning(f"Gradient checkpointing nicht unterstützt für {hf_name}. Training ohne Gradient Checkpointing wird fortgesetzt.")
                        trainer.args.gradient_checkpointing = False
                        trainer.train()
                    else:
                        raise

                results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                logging.info(f"Testergebnisse für {hf_name} ({short_name}): {results}")

                log_experiment(config, hf_name, results, config['Paths']['output'])

                logging.info(f"Trainings- und Evaluationsergebnisse für {hf_name} ({short_name}):")
                for key, value in results.items():
                    logging.info(f"{key}: {value}")

                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)

                logging.info(f"Ausführungszeit für Modell {hf_name} ({short_name}): {(time.time() - total_start_time) / 60:.2f} Minuten")

            update_config_checksum(config, current_checksum)
        else:
            logging.info("Keine Änderungen erkannt. Vervollständige Training für neue Modelle.")
            for hf_name, short_name in models_to_process:
                model_save_path = os.path.join(config['Paths']['models'], short_name)
                if not os.path.exists(model_save_path):
                    logging.info(f"Trainiere neues Modell: {hf_name} ({short_name})")

                    num_labels = len(categories)
                    model, tokenizer = get_model_and_tokenizer(hf_name, num_labels, categories, config)

                    trainer, tokenized_datasets = setup_model_and_trainer(dataset, le, config, hf_name, model, tokenizer, quick=args.quick)
                    trainer.train()
                    results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                    logging.info(f"Testergebnisse für {hf_name} ({short_name}): {results}")

                    log_experiment(config, hf_name, results, config['Paths']['output'])

                    logging.info(f"Trainings- und Evaluationsergebnisse für {hf_name} ({short_name}):")
                    for key, value in results.items():
                        logging.info(f"{key}: {value}")

                    trainer.save_model(model_save_path)
                    tokenizer.save_pretrained(model_save_path)

                    logging.info(f"Ausführungszeit für Modell {hf_name} ({short_name}): {(time.time() - total_start_time) / 60:.2f} Minuten")

        total_end_time = time.time()
        total_duration = (total_end_time - total_start_time) / 60
        logging.info(f"Gesamtausführungszeit für alle Modelle: {total_duration:.2f} Minuten")

        if args.checkthis:
            models_to_check = get_models_for_task(config, 'check')
            for hf_name, short_name in models_to_check:
                model_save_path = os.path.join(config['Paths']['models'], short_name)
                num_labels = len(categories)
                model, tokenizer = get_model_and_tokenizer(hf_name, num_labels, categories, config)

                # Laden des trainierten Modells
                model.load_state_dict(torch.load(os.path.join(model_save_path, 'pytorch_model.bin')))

                check_folder = config['Paths']['check_this']
                analyze_documents_csv(check_folder, {short_name: (model, tokenizer, le)}, extract_text_from_file)

        if args.predict:
            models_to_check = get_models_for_task(config, 'check')
            if models_to_check:
                hf_name, short_name = models_to_check[0]  # Verwende das erste verfügbare Modell
                model_save_path = os.path.join(config['Paths']['models'], short_name)
                num_labels = len(categories)
                model, tokenizer = get_model_and_tokenizer(hf_name, num_labels, categories, config)

                # Laden des trainierten Modells
                model.load_state_dict(torch.load(os.path.join(model_save_path, 'pytorch_model.bin')))

                result = analyze_new_article(args.predict, model, tokenizer, le, extract_text_from_file)
                if result:
                    print(json.dumps(result, indent=2))
                    logging.info(json.dumps(result, indent=2))
                else:
                    print("Konnte keine Analyse durchführen.")
                    logging.info("Konnte keine Analyse durchführen.")
            else:
                print("Keine Modelle für die Vorhersage verfügbar.")
                logging.info("Keine Modelle für die Vorhersage verfügbar.")

        if training_performed:
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
