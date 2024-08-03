import os
import time
from datetime import date
import logging
import configparser
import argparse
import sys
import json
from colorama import Fore, Style
from datetime import date
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

# Importe aus Ihren eigenen Modulen
from file_utils import check_environment, check_hf_token, check_files, extract_text_from_file
from data_processing import create_dataset
from model_utils import setup_model_and_trainer, get_model_and_tokenizer
from analysis_utils import analyze_new_article

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='classifier.log')

def main():

    logging.info("")
    logging.error("Programmstart")
    logging.error("=============")
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
                tokenizer, trainer, tokenized_datasets = setup_model_and_trainer(dataset_dict, le, config, model_name, quick=args.quick)
                trainer.train()
                results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                logging.info(f"Testergebnisse für {model_name}: {results}")
                logging.info("Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight'] You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.")
                logging.info(f"Trainings- und Evaluationsergebnisse:")
                for epoch_results in results:
                    logging.info(f"{epoch_results}")
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
                logging.info("Modellkategorien:", id2label)

                # Erstellen eines neuen LabelEncoders mit den geladenen Kategorien
                le = LabelEncoder()
                le.classes_ = np.array(list(label2id.keys()))

                trainer = Trainer(model=model)

                if args.checkthis:
                    check_files(trainer, tokenizer, le, config, model_name)

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
