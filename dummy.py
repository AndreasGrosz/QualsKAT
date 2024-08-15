if args.checkthis:
    models_to_process = get_models_for_task(config, 'check')
    for hf_name, short_name in models_to_process:
        # Hier fehlt die Definition von model_save_path
        num_labels = len(categories)
        model, tokenizer = get_model_and_tokenizer(hf_name, num_labels, categories, config)

        logging.info(f"Versuche Modell zu laden von: {model_save_path}")
        # Rest des Codes...
