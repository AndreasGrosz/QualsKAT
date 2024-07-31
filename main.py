import os
import time
import logging
import configparser
import argparse
import sys
import json
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
from model_utils import setup_model_and_trainer
from analysis_utils import analyze_new_article


def main():
    parser = argparse.ArgumentParser(
        description='LRH Document Classifier',
        formatter_class=argparse.RawTextHelpFormatter  # Erlaubt Zeilenumbrüche in der Beschreibung
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

        dataset, le = create_dataset(config, quick=args.quick)
        print("Verarbeitete Dokumente:")
        for item in dataset:
            print(f"  - {item['filename']}")

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
            start_time = time.time()

            model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))

            if args.train:
                trainer.train()
                results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                logging.info(f"Testergebnisse für {model_name}: {results}")

                # Speichere das Modell
                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                trainer.model.config.save_pretrained(model_save_path)

                logging.info(f"Modell gespeichert in: {model_save_path}")
                logging.info(f"Inhalt des Modellverzeichnisses:")
                for file in os.listdir(model_save_path):
                    logging.info(f" - {file}")

                end_time = time.time()
                logging.info(f"Trainingszeit für {model_name}: {(end_time - start_time) / 60:.2f} Minuten")

            if args.checkthis or args.predict:
                logging.info(f"Lade Modell für Analyse: {model_name}")
                if not os.path.exists(os.path.join(model_save_path, 'config.json')):
                    logging.error(f"config.json nicht gefunden in {model_save_path}")
                    logging.info("Versuche, das Modell neu zu initialisieren...")
                    tokenizer, model = get_model_and_tokenizer(model_name, len(le.classes_))
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_datasets['train'],
                        eval_dataset=tokenized_datasets['test'],
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics
                    )
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_datasets['train'],
                        eval_dataset=tokenized_datasets['test'],
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics
                    )
                    print("Überprüfe die Struktur des Eval-Datensatzes im Trainer:")
                    print(trainer.eval_dataset.features)
                if args.checkthis:
                    check_files(trainer, tokenizer, le, config)

                if args.predict:
                    result = analyze_new_article(args.predict, trainer, tokenizer, le)
                    if result:
                        print(json.dumps(result, indent=2))
                    else:
                        print("Konnte keine Analyse durchführen.")

        total_end_time = time.time()
        total_duration = (total_end_time - total_start_time) / 60
        logging.info(f"Gesamtausführungszeit für alle Modelle: {total_duration:.2f} Minuten")

    except Exception as e:
        logging.error(f"Ein kritischer Fehler ist aufgetreten: {str(e)}")
        raise

if __name__ == "__main__":
    main()
