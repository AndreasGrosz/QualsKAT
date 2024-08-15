if args.checkthis:
    models_to_process = get_models_for_task(config, 'check')
    for hf_name, short_name in models_to_process:
        model_save_path = os.path.join(config['Paths']['models'], hf_name.replace('/', '_'))
        logging.info(f"Versuche Modell zu laden von: {model_save_path}")

        if not os.path.exists(model_save_path):
            logging.warning(f"Modellverzeichnis nicht gefunden: {model_save_path}")
            continue

        num_labels = len(categories)
        model, tokenizer = get_model_and_tokenizer(hf_name, num_labels, categories, config)

        model_file = os.path.join(model_save_path, 'model.safetensors')
        if not os.path.exists(model_file):
            logging.warning(f"Modelldatei nicht gefunden: {model_file}")
            continue

        try:
            from safetensors.torch import load_file
            model.load_state_dict(load_file(model_file))
            logging.info(f"Modell erfolgreich geladen: {hf_name}")
        except Exception as e:
            logging.error(f"Fehler beim Laden des Modells {hf_name}: {str(e)}")
            continue

        # Hier den Code für die Analyse mit dem geladenen Modell einfügen
        check_files(config, {short_name: (model, tokenizer, le)}, extract_text_from_file)

        # Speicher freigeben
        del model
        torch.cuda.empty_cache()
