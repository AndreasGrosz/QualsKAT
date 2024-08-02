def analyze_new_article(file_path, trainer, tokenizer, label2id, id2label, extract_text_from_file):
    # ... (bestehender Code)

    try:
        top_predictions, lrh_probability, ghostwriter_probability = predict_top_n(trainer, tokenizer, text, label2id, id2label, n=len(label2id))
    except KeyError as e:
        logging.error(f"Unbekannte Kategorie in der Vorhersage für {file_path}: {str(e)}")
        # Verwende eine Standardkategorie oder überspringe den Eintrag
        lrh_probability = "Nicht verfügbar"
        ghostwriter_probability = "Nicht verfügbar"
        top_predictions = []

    # ... (restlicher Code)
