from datetime import datetime
import warnings
from data_processing import extract_text_from_file
import datetime
import torch
import numpy as np
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

warnings.filterwarnings("ignore", message="Some weights of")

def get_model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def setup_model_and_trainer(dataset_dict, le, config, model_name, quick=False):
    num_labels = len(le.classes_)
    tokenizer, model = get_model_and_tokenizer(model_name, num_labels)

    # Speichern Sie die Kategorie-zu-Label-Zuordnung in der Modellkonfiguration
    model.config.label2id = {label: i for i, label in enumerate(le.classes_)}
    model.config.id2label = {i: label for i, label in enumerate(le.classes_)}

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length")
        tokenized["labels"] = examples["labels"]
        return tokenized

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=dataset_dict["train"].column_names)
    model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
    os.makedirs(model_save_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=float(config['Training']['learning_rate']),
        per_device_train_batch_size=int(config['Training']['batch_size']),
        per_device_eval_batch_size=int(config['Training']['batch_size']),
        num_train_epochs=1 if quick else int(config['Training']['num_epochs']),
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    return tokenizer, trainer, tokenized_datasets

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).astype(np.float32).mean().item()}

def predict_top_n(trainer, tokenizer, text, le, n=None):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = trainer.model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    results = []
    for idx, prob in enumerate(probabilities):
        category = le.inverse_transform([idx])[0]
        results.append((category, prob.item()))

    if n is None:
        n = len(le.classes_)

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:n]

    lrh_probability = sum(prob for cat, prob in results if cat.startswith("LRH"))
    ghostwriter_probability = sum(prob for cat, prob in results if cat.startswith("Ghostwriter"))
    unknown_probability = sum(prob for cat, prob in results if cat == "unknown")

    total_prob = lrh_probability + ghostwriter_probability + unknown_probability
    if total_prob > 0:
        lrh_probability /= total_prob
        ghostwriter_probability /= total_prob
        unknown_probability /= total_prob

    return sorted_results, lrh_probability, ghostwriter_probability, unknown_probability


def analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file):
    text = extract_text_from_file(file_path)

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    file_size = len(text.encode('utf-8'))
    top_predictions = predict_top_n(trainer, tokenizer, text, le, n=len(le.classes_))
    
    print("Debug - top_predictions structure:", top_predictions)
    
    # Angepasste Berechnung der Wahrscheinlichkeiten
    if isinstance(top_predictions, dict):
        lrh_probability = sum(conf for cat, conf in top_predictions.items() if cat.startswith("LRH"))
        ghostwriter_probability = sum(conf for cat, conf in top_predictions.items() if cat.startswith("Ghostwriter"))
    elif isinstance(top_predictions, list):
        if all(isinstance(item, tuple) and len(item) == 2 for item in top_predictions):
            lrh_probability = sum(conf for cat, conf in top_predictions if cat.startswith("LRH"))
            ghostwriter_probability = sum(conf for cat, conf in top_predictions if cat.startswith("Ghostwriter"))
        else:
            logging.error(f"Unerwartetes Format für top_predictions: {top_predictions}")
            return None
    else:
        logging.error(f"Unerwartetes Format für top_predictions: {top_predictions}")
        return None

    print(f"Debug - Vorhersagen für {os.path.basename(file_path)}:")
    for category, prob in (top_predictions.items() if isinstance(top_predictions, dict) else top_predictions)[:5]:
        print(f"{category}: {prob:.4f}")

    print(f"LRH Gesamtwahrscheinlichkeit: {lrh_probability:.4f}")
    print(f"Ghostwriter Gesamtwahrscheinlichkeit: {ghostwriter_probability:.4f}")

    threshold = 0.1  # 10% Unterschied als Schwellenwert
    if abs(lrh_probability - ghostwriter_probability) < threshold:
        conclusion = "Nicht eindeutig"
    elif lrh_probability > ghostwriter_probability:
        conclusion = "Wahrscheinlich LRH"
    else:
        conclusion = "Wahrscheinlich Ghostwriter"

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße": file_size,
        "Datum": datetime.now().strftime("%d-%m-%y %H:%M"),
        "LRH": f"{lrh_probability:.2f}",
        "Ghostwriter": f"{ghostwriter_probability:.2f}",
        "Schlussfolgerung": conclusion
    }
