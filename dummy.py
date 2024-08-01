def main():
    # ... (vorheriger Code bleibt unverändert)

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

    for model_name in model_names:
        model_name = model_name.strip()
        logging.info(f"Verarbeite Modell: {model_name}")

        tokenizer, trainer, tokenized_datasets = setup_model_and_trainer(dataset_dict, len(le.classes_), config, model_name, quick=args.quick)

        if args.train:
            logging.info(f"Trainiere Modell: {model_name}")
            trainer.train()
            results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
            logging.info(f"Testergebnisse für {model_name}: {results}")

            # Speichere das Modell
            model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)

        # ... (restlicher Code für checkthis und predict bleibt unverändert)

    # ... (restlicher Code bleibt unverändert)
