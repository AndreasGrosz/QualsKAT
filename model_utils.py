# In model_utils.py
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# Funktionsdefinitionen hier...

def get_model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def setup_model_and_trainer(dataset_dict, num_labels, config, model_name):
    tokenizer, model = get_model_and_tokenizer(model_name, num_labels)

    model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
    os.makedirs(model_save_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=float(config['Training']['learning_rate']),
        per_device_train_batch_size=int(config['Training']['batch_size']),
        per_device_eval_batch_size=int(config['Training']['batch_size']),
        num_train_epochs=int(config['Training']['num_epochs']),
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    if 'validation' not in dataset_dict:
        logging.warning("Kein Validierungsdatensatz gefunden. Verwende Testdatensatz für die Validierung.")
        eval_dataset = dataset_dict['test']
    else:
        eval_dataset = dataset_dict['validation']

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'] if 'validation' in tokenized_datasets else tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    return tokenizer, trainer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).astype(np.float32).mean().item()}

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

    if text is None or len(text) == 0:
        logging.warning(f"Konnte Text aus {file_path} nicht extrahieren oder Text ist leer.")
        return None

    file_size = len(text.encode('utf-8'))
    top_predictions = predict_top_n(trainer, tokenizer, text, le, n=5)

    lrh_probability = sum(conf for cat, conf in top_predictions if cat.startswith("LRH"))
    ghostwriter_probability = sum(conf for cat, conf in top_predictions if cat.startswith("Ghostwriter"))

    threshold = 0.1  # 10% Unterschied als Schwellenwert
    if abs(lrh_probability - ghostwriter_probability) < threshold:
        conclusion = "Nicht eindeutig"
    elif lrh_probability > ghostwriter_probability:
        conclusion = "Wahrscheinlich LRH"
    else:
        conclusion = "Wahrscheinlich Ghostwriter"

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße (Bytes)": file_size,
        "Datum": date.today().strftime("%Y-%m-%d"),
        "LRH Wahrscheinlichkeit": f"{lrh_probability:.2f}",
        "Ghostwriter Wahrscheinlichkeit": f"{ghostwriter_probability:.2f}",
        "Schlussfolgerung": conclusion
    }

