def main():
    # ... (vorheriger Code bleibt unverändert)

    # Überprüfen Sie, ob die Kategorien-Datei existiert
    categories_file = os.path.join(config['Paths']['output'], 'categories.csv')
    if not os.path.exists(categories_file):
        print("Kategorien-Datei nicht gefunden. Bitte führen Sie zuerst update_categories.py aus.")
        sys.exit(1)

    dataset, le = create_dataset(config, quick=args.quick)

    # ... (restlicher Code bleibt unverändert)
