import os
import torch
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime

# Import your custom modules
from file_utils import check_environment, check_hf_token, check_files, extract_text_from_file
from data_processing import create_dataset
from model_utils import setup_model_and_trainer, get_model_and_tokenizer, predict_top_n

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='classifier.log')

def main():
    logging.info("Program started")

    parser = argparse.ArgumentParser(description='LRH Document Classifier')
    parser.add_argument('--train', action='store_true', help='Train the model with documents in the "documents" folder.')
    parser.add_argument('--checkthis', action='store_true', help='Analyze files in the "CheckThis" folder with a trained model.')
    parser.add_argument('--quick', action='store_true', help='Perform a quick analysis with a fraction of the data.')
    parser.add_argument('--predict', metavar='FILE', help='Make a prediction for a single file.')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        config = check_environment()
        check_hf_token()

        categories_file = os.path.join(config['Paths']['output'], 'categories.csv')
        if not os.path.exists(categories_file):
            print("Categories file not found. Please run update_categories.py first.")
            logging.error("Categories file not found. Please run update_categories.py first.")
            sys.exit(1)

        dataset, le = create_dataset(config, quick=args.quick)
        if len(dataset) == 0:
            raise ValueError("The created dataset is empty. Check the input data.")

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
            logging.info(f"Processing model: {model_name}")

            model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))

            if args.train:
                tokenizer, trainer, tokenized_datasets = setup_model_and_trainer(dataset_dict, le, config, model_name, quick=args.quick)
                trainer.train()
                results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                logging.info(f"Test results for {model_name}: {results}")
                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)

            if args.checkthis or args.predict:
                model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)

                label2id = model.config.label2id
                id2label = model.config.id2label
                print("Model categories:", id2label)
                logging.info("Model categories:", id2label)

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
                        print("Could not perform analysis.")
                        logging.info("Could not perform analysis.")

        total_end_time = time.time()
        total_duration = (total_end_time - total_start_time) / 60
        logging.info(f"Total execution time for all models: {total_duration:.2f} minutes")

    except Exception as e:
        logging.error(f"A critical error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
