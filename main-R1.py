import argparse
import time
from datetime import datetime
import os
import olefile
import csv
import logging
import configparser
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from striprtf.striprtf import rtf_to_text
import docx
import pdfplumber
from analysis_utils import analyze_new_article
import file_utils_ag
from file_utils_ag import check_environment, extract_text_from_file, get_device, check_files
from data_processing import load_categories_from_csv
from sklearn.preprocessing import LabelEncoder
from OCR_Error_eval import main as ocr_eval_main


def load_label_encoder(config):
    categories = load_categories_from_csv(config)
    if len(categories) != 2 or 'LRH' not in categories or 'Nicht-LRH' not in categories:
        raise ValueError("Die categories.csv Datei muss genau zwei Zeilen enthalten: 'Nicht-LRH' und 'LRH'")
    le = LabelEncoder()
    le.fit(categories)
    print("Label-Encoder Klassen:", le.classes_)
    return le


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


def main():
    parser = argparse.ArgumentParser(description="Hauptprogramm für Textanalyse")
    parser.add_argument("--ocr", action="store_true", help="Führe OCR-Fehlererkennung durch")
    parser.add_argument("--scn_words", default="preproc/ScnWortListe/ScnWorte.txt", help="Pfad zur Scientology-Wortliste")
    parser.add_argument("thema", help="Thema oder Bezeichnung für diesen Analysedurchlauf")
    parser.add_argument("--checkthis", action="store_true", help="Führe die Analyse für den 'check_this' Ordner durch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.ocr:
        ocr_eval_main(args)
    else:

        try:
            check_hf_token()
            config, models, device = check_environment()

            le = load_label_encoder(config)

            timestamp = datetime.now().strftime("%y%m%d-%Hh%M")

            for model_name, (tokenizer, model) in models.items():
                model.to(device)
                logging.info(f"Verarbeite Modell: {model_name}")

                if args.checkthis:
                    output_filename = f"{timestamp}-Results_{args.thema}_{model_name}.csv"
                    output_path = os.path.join(config['Paths']['output'], output_filename)
                    print(f"Debug: Calling check_files with {model_name} and {output_path}")
                    check_files(model, tokenizer, le, config, model_name, output_path)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            logging.info("Alle Modelle wurden verarbeitet.")

        except Exception as e:
            logging.error(f"Ein kritischer Fehler ist aufgetreten: {str(e)}")
            raise

if __name__ == "__main__":
    main()


