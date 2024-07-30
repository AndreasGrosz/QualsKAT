import os
import argparse
from pathlib import Path
import time
import csv
from datetime import date
import logging
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

def get_project_root(is_colab):
    if is_colab:
        from google.colab import drive
        drive.mount('/content/drive')
        return Path('/content/drive/MyDrive/QualsKAT')
    else:
        return Path(__file__).parent

def setup_paths(is_colab):
    root = get_project_root(is_colab)
    return {
        'documents': root / 'Documents',
        'check_this': root / 'CheckThis',
        'output': root / 'output',
        'models': root / 'models'
    }

def check_hf_token():
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token is None:
        raise ValueError("Hugging Face API token is missing. Please set the environment variable 'HUGGINGFACE_HUB_TOKEN'.")

    api = HfApi()
    try:
        api.whoami(token=hf_token)
        print("Hugging Face API token is valid.")
    except HTTPError as e:
        raise ValueError("Invalid Hugging Face API token. Please check the token and try again.")

def extract_text_from_file(file_path):
    # Implementierung hier (Ihre bestehende Funktion)
    pass

def get_files_and_categories(paths, test_mode=False):
    root_dir = paths['documents']
    files_and_categories = []
    supported_formats = ['.txt', '.doc', '.docx', '.rtf', '.pdf', '.html', '.htm']

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(fmt) for fmt in supported_formats):
                file_path = os.path.join(root, file)
                categories = os.path.relpath(root, root_dir).split(os.sep)

                if categories[0] == "LRH":
                    category = f"LRH-{'-'.join(categories[1:])}"
                elif categories[0] == "unknown":
                    category = "unknown"
                else:
                    category = f"Ghostwriter-{categories[0]}"

                files_and_categories.append((file_path, category))

                if test_mode and len(files_and_categories) >= 10:
                    return files_and_categories
    return files_and_categories

def create_dataset(paths, test_mode=False):
    files_and_categories = get_files_and_categories(paths, test_mode)
    texts = []
    all_categories = []

    for file_path, category in files_and_categories:
        text = extract_text_from_file(file_path)
        if text:
            if test_mode:
                text = text[:10000]  # Limit to first 10000 characters in test mode
            texts.append(text)
            all_categories.append(category)

    le = LabelEncoder()
    numeric_categories = le.fit_transform(all_categories)

    dataset_dict = {
        'text': texts,
        'labels': numeric_categories
    }

    return Dataset.from_dict(dataset_dict), le

def collect_category_sizes(paths):
    root_dir = paths['documents']
    category_sizes = {}
    files_and_categories = get_files_and_categories(paths)

    for file_path, category in files_and_categories:
        text = extract_text_from_file(file_path)
        if text:
            size = len(text.encode('utf-8'))  # Größe in Bytes
            if category in category_sizes:
                category_sizes[category] += size
            else:
                category_sizes[category] = size

    return category_sizes

def print_category_sizes(category_sizes):
    print("Kategorien und ihre Größen:")

    lrh_total = 0
    ghostwriter_total = 0
    unknown_total = 0

    for category, size in sorted(category_sizes.items()):
        print(f"{category}: {size} Bytes")
        if category.startswith("LRH"):
            lrh_total += size
        elif category.startswith("Ghostwriter"):
            ghostwriter_total += size
        elif category == "unknown":
            unknown_total += size

    print("\nZusammenfassung:")
    print(f"LRH Gesamt: {lrh_total} Bytes")
    print(f"Ghostwriter Gesamt: {ghostwriter_total} Bytes")
    print(f"Unbekannt Gesamt: {unknown_total} Bytes")
    print(f"Gesamtzahl der Kategorien: {len(category_sizes)}")
    print(f"Gesamtgröße aller Kategorien: {sum(category_sizes.values())} Bytes")
    print()

def setup_model_and_trainer(dataset_dict, num_labels, paths):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=str(paths['models']),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    return tokenizer, trainer

def train_and_evaluate(trainer, tokenized_dataset, paths):
    trainer.train()
    results = trainer.evaluate(tokenized_dataset['test'])
    print("Test results:", results)
    trainer.save_model(str(paths['models'] / "Final_Model"))

def predict_top_n(trainer, tokenizer, text, le, n=3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = trainer.model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_n_prob, top_n_indices = torch.topk(probabilities, n)
    results = []
    for prob, idx in zip(top_n_prob[0], top_n_indices[0]):
        category = le.inverse_transform([idx.item()])[0]
        results.append((category, prob.item()))
    return results

def analyze_new_article(file_path, trainer, tokenizer, le):
    text = extract_text_from_file(file_path)

    if text is None:
        return None

    file_size = len(text.encode('utf-8'))
    top_predictions = predict_top_n(trainer, tokenizer, text, le, n=5)

    lrh_probability = sum(conf for cat, conf in top_predictions if cat.startswith("LRH"))
    ghostwriter_probability = sum(conf for cat, conf in top_predictions if cat.startswith("Ghostwriter"))

    conclusion = "Wahrscheinlich LRH" if lrh_probability > ghostwriter_probability else "Wahrscheinlich nicht LRH"

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße (Bytes)": file_size,
        "Datum": date.today().strftime("%Y-%m-%d"),
        "LRH Wahrscheinlichkeit": f"{lrh_probability:.2f}",
        "Ghostwriter Wahrscheinlichkeit": f"{ghostwriter_probability:.2f}",
        "Schlussfolgerung": conclusion
    }

def check_files(trainer, tokenizer, le, paths):
    check_folder = paths['check_this']
    if not os.path.exists(check_folder):
        print(f"Der Ordner '{check_folder}' existiert nicht.")
        return

    files = [f for f in os.listdir(check_folder) if os.path.isfile(os.path.join(check_folder, f))]

    if not files:
        print(f"Keine Dateien im Ordner '{check_folder}' gefunden.")
        return

    results = []
    for file in files:
        file_path = os.path.join(check_folder, file)
        result = analyze_new_article(file_path, trainer, tokenizer, le)
        if result:
            results.append(result)

    csv_filename = paths['output'] / "CheckThisResults.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Ergebnisse wurden in {csv_filename} gespeichert.")

    lrh_count = sum(1 for r in results if r["Schlussfolgerung"] == "Wahrscheinlich LRH")
    print(f"\nZusammenfassung: {len(results)} Dateien analysiert, {lrh_count} wahrscheinlich von LRH.")

def compute_metrics(eval_pred):
    # Implementierung hier
    pass

def main():
    parser = argparse.ArgumentParser(description='LRH Document Classifier')
    parser.add_argument('--local', action='store_true', help='Run in local mode')
    parser.add_argument('--approach', choices=['discriminative'], default='discriminative')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--checkthis', action='store_true')
    args = parser.parse_args()

    is_colab = not args.local
    paths = setup_paths(is_colab)

    start_time = time.time()
    print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    check_hf_token()

    dataset, le = create_dataset(paths, args.test_mode)
    dataset_dict = dataset.train_test_split(test_size=0.2)

    tokenizer, trainer = setup_model_and_trainer(dataset_dict, len(le.classes_), paths)

    if not args.checkthis:
        category_sizes = collect_category_sizes(paths)
        print_category_sizes(category_sizes)
        train_and_evaluate(trainer, dataset_dict, paths)
    else:
        check_files(trainer, tokenizer, le, paths)

    end_time = time.time()
    print(f"Script ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
