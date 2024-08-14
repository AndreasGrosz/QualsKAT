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
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, set_seed, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, hf_hub_download
from requests.exceptions import HTTPError

# Importe aus Ihren eigenen Modulen
from file_utils import check_environment, check_hf_token, check_files, extract_text_from_file, calculate_documents_checksum, update_config_checksum
from data_processing import create_dataset, load_categories_from_csv
from model_utils import setup_model_and_trainer, get_model_and_tokenizer, get_models_for_task, get_optimizer_and_scheduler

from analysis_utils import analyze_new_article, analyze_document, analyze_documents_csv
from file_utils import get_device, extract_text_from_file
from experiment_logger import log_experiment
from colorama import Fore, Back, Style, init
import warnings


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")

init(autoreset=True)  # Initialisiert Colorama


if hasattr(torch.cuda.amp, 'GradScaler'):
    torch.cuda.amp.GradScaler = lambda **kwargs: torch.amp.GradScaler('cuda', **kwargs)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='classifier.log')


if torch.cuda.is_available():
    print(f"Verfügbarer GPU-Speicher: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def get_total_steps(trainer):
    return len(trainer.get_train_dataloader())


def print_training_progress(epoch, step, total_steps, loss, grad_norm, learning_rate, start_time):
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / (step + 1) * total_steps
    remaining_time = estimated_total_time - elapsed_time

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}Epoch: {epoch:.2f} | Step: {step}/{total_steps}")
    print(f"{Fore.GREEN}Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f}")
    print(f"{Fore.BLUE}Learning Rate: {learning_rate:.6f}")
    print(f"{Fore.MAGENTA}Elapsed Time: {elapsed_time/60:.2f} min | Remaining Time: {remaining_time/60:.2f} min")
    print(f"{Fore.CYAN}{'='*60}\n")


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

            for hf_name, short_name in models_to_process:
                print(f"\n{Fore.YELLOW}{'*'*80}")
                print(f"{Fore.YELLOW}Training Model: {hf_name} ({short_name})")
                print(f"{Fore.YELLOW}{'*'*80}\n")

                model_save_path = os.path.join(config['Paths']['models'], short_name)
                num_labels = len(categories)
                model, tokenizer = get_model_and_tokenizer(hf_name, num_labels, categories, config)

                trainer, tokenized_datasets = setup_model_and_trainer(dataset, le, config, hf_name, model, tokenizer, args.quick)

                total_steps = get_total_steps(trainer)

                start_time = time.time()
                total_steps = len(trainer.get_train_dataloader()) * int(config['Training']['num_epochs'])

                try:
                    print(f"\n{Fore.CYAN}Starte Training für {hf_name}")
                    train_dataloader = trainer.get_train_dataloader()
                    total_steps = len(train_dataloader) * int(config['Training']['num_epochs'])

                    for epoch in range(int(config['Training']['num_epochs'])):
                        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
                        for step, batch in enumerate(progress_bar):
                            loss = trainer.training_step(model, batch)
                            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                            if step % 10 == 0:  # Print every 10 steps
                                global_step = epoch * len(train_dataloader) + step
                                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.max_grad_norm).item()
                                lr = trainer.lr_scheduler.get_last_lr()[0] if trainer.lr_scheduler else trainer.args.learning_rate
                                print_training_progress(
                                    epoch + step/len(train_dataloader),
                                    global_step,
                                    total_steps,
                                    loss.item(),
                                    grad_norm,
                                    lr,
                                    start_time
                                )

                        # Evaluation am Ende jeder Epoche
                        eval_results = trainer.evaluate()
                        print(f"\n{Fore.GREEN}Evaluationsergebnisse nach Epoche {epoch+1}:")
                        for key, value in eval_results.items():
                            print(f"{Fore.CYAN}{key}: {value}")

                    print(f"\n{Fore.GREEN}Training abgeschlossen.")
                except Exception as e:
                    print(f"{Fore.RED}Fehler während des Trainings: {str(e)}")
                    print(f"{Fore.YELLOW}Versuche, das Training zu beenden...")
                    if hasattr(trainer, 'state'):
                        trainer.state.global_step = total_steps  # Force training to end
                    if hasattr(trainer, 'is_in_train'):
                        trainer.is_in_train = False

                print(f"\n{Fore.CYAN}Trainings-Zusammenfassung für {hf_name}:")
                print(f"{Fore.CYAN}Anzahl der Epochen: {config['Training']['num_epochs']}")
                print(f"{Fore.CYAN}Anzahl der Batches pro Epoche: {total_steps}")
                print(f"{Fore.CYAN}Gesamtanzahl der Batches: {int(config['Training']['num_epochs']) * total_steps}")
                print(f"{Fore.CYAN}Batch-Größe: {config['Training']['batch_size']}")
                print(f"{Fore.CYAN}Gesamtanzahl der Trainingsdokumente: {len(tokenized_datasets['train'])}")
                print(f"{Fore.CYAN}Geschätzte Trainingszeit: {(int(config['Training']['num_epochs']) * total_steps * 4) / 60:.2f} Minuten")
                print(f"{Fore.CYAN}{'='*60}\n")

                start_time = time.time()

                try:
                    trainer.train()
                except Exception as e:
                    print(f"Fehler während des Trainings: {str(e)}")
                    print("Versuche, das Training zu beenden...")
                    trainer.state.global_step = trainer.args.max_steps  # Force training to end
                    trainer.is_in_train = False

                total_steps = get_total_steps(trainer)
                start_time = time.time()

                try:
                    print(f"\n{Fore.CYAN}Starte Training für {hf_name}")
                    train_dataloader = trainer.get_train_dataloader()
                    total_steps = len(train_dataloader) * int(config['Training']['num_epochs'])

                    for epoch in range(int(config['Training']['num_epochs'])):
                        progress_bar = tqdm(trainer.get_train_dataloader(), desc=f"Epoch {epoch+1}")
                        for step, batch in enumerate(progress_bar):
                            loss = trainer.training_step(model, batch)
                            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                            if step % 10 == 0:  # Print every 10 steps
                                global_step = epoch * len(train_dataloader) + step
                                print_training_progress(
                                    epoch + step/len(train_dataloader),
                                    global_step,
                                    total_steps,
                                    loss.item(),
                                    model.named_parameters().grad.norm().item(),
                                    trainer.lr_scheduler.get_last_lr()[0],
                                    start_time
                                )

                        # Evaluation am Ende jeder Epoche
                        eval_results = trainer.evaluate()
                        print(f"\n{Fore.GREEN}Evaluationsergebnisse nach Epoche {epoch+1}:")
                        for key, value in eval_results.items():
                            print(f"{Fore.CYAN}{key}: {value}")

                    print(f"\n{Fore.GREEN}Training abgeschlossen.")
                except Exception as e:
                    print(f"{Fore.RED}Fehler während des Trainings: {str(e)}")
                    print(f"{Fore.YELLOW}Versuche, das Training zu beenden...")
                    if hasattr(trainer, 'state'):
                        trainer.state.global_step = total_steps  # Force training to end
                    if hasattr(trainer, 'is_in_train'):
                        trainer.is_in_train = False

                    else:
                        raise

                results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                print(f"\n{Fore.GREEN}Testergebnisse für {hf_name} ({short_name}):")
                for key, value in results.items():
                    print(f"{Fore.CYAN}{key}: {value}")

                log_experiment(config, hf_name, results, config['Paths']['output'])

                for param in trainer.model.parameters():
                    if not param.data.is_contiguous():
                        param.data = param.data.contiguous()

                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)

                print(f"\n{Fore.MAGENTA}Ausführungszeit für Modell {hf_name} ({short_name}): {(time.time() - start_time) / 60:.2f} Minuten")

            update_config_checksum(config, current_checksum)


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

                for param in trainer.model.parameters():
                    if not param.data.is_contiguous():
                        param.data = param.data.contiguous()

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

        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Training abgeschlossen für {hf_name}")
        print(f"{Fore.YELLOW}Finale Loss: {trainer.state.log_history[-1]['loss']}")
        print(f"{Fore.YELLOW}Anzahl der durchgeführten Schritte: {trainer.state.global_step}")
        print(f"{Fore.CYAN}{'='*80}")
        logging.error("---------------------------------------")
        logging.error("Programmende")

    except Exception as e:
        logging.error(f"Ein kritischer Fehler ist aufgetreten: {str(e)}")
        raise

if __name__ == "__main__":
    main()
