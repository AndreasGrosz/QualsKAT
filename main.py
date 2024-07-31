import argparse
import logging
import time
import os
from data_processing import create_dataset
from model_utils import setup_model_and_trainer
from datasets import DatasetDict
from file_utils import check_environment, check_hf_token, check_files, extract_text_from_file

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
            logging.info(f"Modell {model_name} erfolgreich geladen und Trainer eingerichtet.")

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
                check_files(trainer, tokenizer, le, config)

        total_end_time = time.time()
        total_duration = (total_end_time - total_start_time) / 60
        logging.info(f"Total execution time for all models: {total_duration:.2f} minutes")

    except Exception as e:
        logging.error(f"Ein kritischer Fehler ist aufgetreten: {str(e)}")
        raise

if __name__ == "__main__":
    main()
