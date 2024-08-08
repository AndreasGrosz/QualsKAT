from datetime import datetime
import warnings
from data_processing import extract_text_from_file
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from accelerate import Accelerator


warnings.filterwarnings("ignore", message="Some weights of")


def get_model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_optimal_batch_size(model, max_sequence_length, device):
    if device.type == "cuda":
        mem = torch.cuda.get_device_properties(device).total_memory
        model_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        sequence_mem = model.config.hidden_size * max_sequence_length * 4  # 4 bytes per float
        max_batch_size = (mem - model_mem) // (sequence_mem * 2)  # Factor of 2 for safety
        return max(1, min(64, max_batch_size))  # Cap at 64 for stability
    else:
        return 8  # Ein vernünftiger Standardwert für CPUs


from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding


def setup_model_and_trainer(dataset_dict, le, config, model_name, quick=False):
    # ... (vorheriger Code bleibt unverändert)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=float(config['Training']['learning_rate']),
        per_device_train_batch_size=4,  # Reduzierte Batch-Größe
        per_device_eval_batch_size=4,   # Reduzierte Batch-Größe
        num_train_epochs=1 if quick else int(config['Training']['num_epochs']),
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,  # Aktiviert Mixed Precision Training
        gradient_accumulation_steps=4,  # Gradient Accumulation
        gradient_checkpointing=True,  # Aktiviert Gradient Checkpointing
        logging_dir=os.path.join(model_save_path, 'logs'),
        logging_steps=100,
        save_total_limit=2,  # Behält nur die besten 2 Checkpoints
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return tokenizer, trainer, dataset_dict



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).astype(np.float32).mean().item()}


def predict_top_n(trainer, tokenizer, text, le, n=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    trainer.model.to(device)
    with torch.no_grad():
        logits = trainer.model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    results = []
    for idx, prob in enumerate(probabilities):
        category = le.inverse_transform([idx])[0]
        results.append((category, prob.item()))

    if n is None:
        n = len(le.classes_)

    return sorted(results, key=lambda x: x[1], reverse=True)[:n]


def analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file):
    text = extract_text_from_file(file_path)

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    file_size = len(text.encode('utf-8'))
    top_predictions = predict_top_n(trainer, tokenizer, text, le, n=len(le.classes_))
    
    print("Debug - top_predictions structure:", top_predictions)
    logging.info("Debug - top_predictions structure:", top_predictions)
    
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

    print(f"Vorhersagen für {os.path.basename(file_path)}:")
    loggin.info(f"Vorhersagen für {os.path.basename(file_path)}:")
    for category, prob in (top_predictions.items() if isinstance(top_predictions, dict) else top_predictions)[:5]:
        print(f"{category}: {prob:.4f}")
        loggin.info(f"{category}: {prob:.4f}")

    print(f"LRH Gesamtwahrscheinlichkeit: {lrh_probability:.4f}")
    print(f"Ghostwriter Gesamtwahrscheinlichkeit: {ghostwriter_probability:.4f}")
    loggin.info(f"LRH Gesamtwahrscheinlichkeit: {lrh_probability:.4f}")
    loggin.info(f"Ghostwriter Gesamtwahrscheinlichkeit: {ghostwriter_probability:.4f}")

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
