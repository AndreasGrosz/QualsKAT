def analyze_documents_csv(check_folder, models, extract_text_from_file):
    files = [f for f in os.listdir(check_folder) if os.path.isfile(os.path.join(check_folder, f))]

    # CSV-Header
    header = ['Filename', 'r-base', 'ms-deberta', 'distilb', 'r-large', 'albert', 'Mittelwert']
    print(','.join(header))

    for file in tqdm(files, desc="Analysiere Dateien"):
        file_path = os.path.join(check_folder, file)
        text = extract_text_from_file(file_path)

        if text is None or len(text) == 0:
            print(f'"{file}",N/A,N/A,N/A,N/A,N/A,N/A')
            continue

        results = []
        for model_name, (model, tokenizer, le) in models.items():
            confidence = calculate_lrh_confidence(model, tokenizer, text, le)
            results.append(confidence)

        avg_confidence = mean(results)

        # CSV-Zeile ausgeben
        output = [f'"{file}"'] + [f"{conf:.1f}" for conf in results] + [f"{avg_confidence:.1f}"]
        print(','.join(output))

    sys.stdout.flush()
