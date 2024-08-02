def analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file):
    text = extract_text_from_file(file_path)

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    file_size = len(text.encode('utf-8'))

    # Erstelle einen neuen LabelEncoder und passe ihn an die Kategorien aus dem Modell an
    le = LabelEncoder()
    le.classes_ = np.array(list(trainer.model.config.id2label.values()))

    # ... (restlicher Code)
