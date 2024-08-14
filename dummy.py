def main():
    # ... (vorheriger Code bleibt unverändert)

    if args.train:
        training_performed = True
        models_to_train = get_models_for_task(config, 'train')

        if current_checksum != stored_checksum:
            logging.info("Änderungen im documents-Ordner erkannt. Starte vollständiges Neutraining.")
            for hf_name, short_name in models_to_train:
                logging.info(f"Trainiere Modell: {hf_name} ({short_name})")

                model_save_path = os.path.join(config['Paths']['models'], short_name)

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

            update_config_checksum(config, current_checksum)
        else:
            logging.info("Keine Änderungen erkannt. Vervollständige Training für neue Modelle.")
            for hf_name, short_name in models_to_train:
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

    # ... (Rest des Codes bleibt unverändert)
