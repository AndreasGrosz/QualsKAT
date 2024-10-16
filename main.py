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
from analysis_utils import analyze_new_article, analyze_document, analyze_documents_csv
from file_utils import get_device, extract_text_from_file
from experiment_logger import log_experiment

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_cuda():
    if torch.cuda.is_available():
        logger.info(f"CUDA ist verfügbar. Gefundene Geräte: {torch.cuda.device_count()}")
        logger.info(f"Aktuelles CUDA-Gerät: {torch.cuda.get_device_name(0)}")
        return True
    else:
        logger.warning("CUDA ist nicht verfügbar. Verwende CPU.")
        return False


# Aktualisiere den GradScaler
if hasattr(torch.cuda.amp, 'GradScaler'):
    torch.cuda.amp.GradScaler = lambda **kwargs: torch.amp.GradScaler('cuda', **kwargs)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='AANaL-errors.log')

def main():
    logging.info("")
    logging.error("Programmstart")
    logging.error("=============")
    cuda_available = check_cuda()

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
        config, models, device = check_environment()
        check_hf_token()

        categories = load_categories_from_csv(config)
        if len(categories) == 0:
            raise ValueError("Keine Kategorien gefunden. Bitte überprüfen Sie die categories.csv Datei.")

        dataset, le = create_dataset(config, quick=args.quick)
        if len(dataset) == 0:
            raise ValueError("Der erstellte Datensatz ist leer. Überprüfen Sie die Eingabedaten.")

        model_names = config['Model']['model_name'].split(',')

        total_start_time = time.time()
        training_performed = False

        if args.train:
            training_performed = True
            for model_name in model_names:
                model_name = model_name.strip()
                logging.info(f"Trainiere Modell: {model_name}")

                model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))

                num_labels = len(categories)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label={i: cat for i, cat in enumerate(categories)},
                    label2id={cat: i for i, cat in enumerate(categories)}
                )

                tokenizer, trainer, tokenized_datasets = setup_model_and_trainer(dataset, le, config, model_name, model, quick=args.quick)
                trainer.train()
                results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                logging.info(f"Testergebnisse für {model_name}: {results}")

                log_experiment(config, model_name, results, config['Paths']['output'])

                logging.info(f"Trainings- und Evaluationsergebnisse:")
                for key, value in results.items():
                    logging.info(f"{key}: {value}")
                logging.info(f"Ausführungszeit für Modell {model_name}: {(time.time() - total_start_time) / 60:.2f} Minuten")
                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)

            total_end_time = time.time()
            total_duration = (total_end_time - total_start_time) / 60
            logging.info(f"Gesamtausführungszeit für alle Modelle: {total_duration:.2f} Minuten")

        if args.checkthis:
            models = {
                'r-base': (AutoModelForSequenceClassification.from_pretrained(os.path.join(config['Paths']['models'], 'roberta-base')),
                           AutoTokenizer.from_pretrained(os.path.join(config['Paths']['models'], 'roberta-base')),
                           LabelEncoder().fit(['Nicht-LRH', 'LRH'])),
                'ms-deberta': (AutoModelForSequenceClassification.from_pretrained(os.path.join(config['Paths']['models'], 'microsoft_deberta-base')),
                               AutoTokenizer.from_pretrained(os.path.join(config['Paths']['models'], 'microsoft_deberta-base')),
                               LabelEncoder().fit(['Nicht-LRH', 'LRH'])),
                'distilb': (AutoModelForSequenceClassification.from_pretrained(os.path.join(config['Paths']['models'], 'distilbert-base-uncased')),
                            AutoTokenizer.from_pretrained(os.path.join(config['Paths']['models'], 'distilbert-base-uncased')),
                            LabelEncoder().fit(['Nicht-LRH', 'LRH'])),
                'r-large': (AutoModelForSequenceClassification.from_pretrained(os.path.join(config['Paths']['models'], 'roberta-large')),
                            AutoTokenizer.from_pretrained(os.path.join(config['Paths']['models'], 'roberta-large')),
                            LabelEncoder().fit(['Nicht-LRH', 'LRH'])),
                'albert': (AutoModelForSequenceClassification.from_pretrained(os.path.join(config['Paths']['models'], 'albert-base-v2')),
                           AutoTokenizer.from_pretrained(os.path.join(config['Paths']['models'], 'albert-base-v2')),
                           LabelEncoder().fit(['Nicht-LRH', 'LRH']))
            }

            check_folder = config['Paths']['check_this']
            output_file = os.path.join(config['Paths']['output'], 'analysis_results.csv')
            analyze_documents_csv(check_folder, models, extract_text_from_file, output_file, save_interval=100)

        if args.predict:
            result = analyze_new_article(args.predict, trainer, tokenizer, le, extract_text_from_file)
            if result:
                print(json.dumps(result, indent=2))
                logging.info(json.dumps(result, indent=2))
            else:
                print("Konnte keine Analyse durchführen.")
                logging.info("Konnte keine Analyse durchführen.")

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
