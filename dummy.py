import logging

# ... (bestehender Code)

def main():
    # ... (bestehender Code)

    try:
        # ... (bestehender Code)

        model_names = config['Model']['model_name'].split(',')
        total_start_time = time.time()

        for model_name in model_names:
            model_name = model_name.strip()
            logging.info(f"{Fore.GREEN}Verarbeite Modell: {model_name}{Style.RESET_ALL}")

            model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))

            if args.train:
                tokenizer, trainer, tokenized_datasets = setup_model_and_trainer(dataset_dict, le, config, model_name, quick=args.quick)
                trainer.train()
                results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
                logging.info(f"Testergebnisse für {model_name}: {results}")

                # Füge diese Informationen ins Logfile ein
                logging.info("Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight'] You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.")
                logging.info(f"Trainings- und Evaluationsergebnisse:")
                for epoch_results in results:
                    logging.info(f"{epoch_results}")
                logging.info(f"Gesamtausführungszeit für Modell {model_name}: {(time.time() - total_start_time) / 60:.2f} Minuten")

                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)

            # ... (restlicher Code)
    except Exception as e:
        logging.error(f"Ein kritischer Fehler ist aufgetreten: {str(e)}")
        raise

if __name__ == "__main__":
    main()
