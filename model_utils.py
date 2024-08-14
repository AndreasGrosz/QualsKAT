from datetime import datetime
import warnings
from data_processing import extract_text_from_file
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, LlamaForCausalLM, LlamaTokenizer
import numpy as np
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm


warnings.filterwarnings("ignore", message="Some weights of")

def predict_for_model(model, tokenizer, text, le):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    results = []
    for idx, prob in enumerate(probabilities):
        category = le.inverse_transform([idx])[0]
        results.append((category, prob.item()))

    return sorted(results, key=lambda x: x[1], reverse=True)


def get_model_and_tokenizer(model_name, num_labels, categories, config):
    base_model_path = os.path.join(os.path.dirname(__file__), 'fresh-models')
    local_model_path = os.path.join(base_model_path, model_name)

    if model_name == "Meta-Llama-3-8B":
        # Spezielle Behandlung für das Llama-Modell
        tokenizer = LlamaTokenizer.from_pretrained(local_model_path)
        model = LlamaForCausalLM.from_pretrained(local_model_path)

        # Hier müssen wir möglicherweise das Modell für Sequenzklassifizierung anpassen
        # Dies ist ein Platzhalter und muss wahrscheinlich angepasst werden
        model = AutoModelForSequenceClassification.from_pretrained(
            model.config._name_or_path,
            num_labels=num_labels,
            id2label={i: label for i, label in enumerate(categories)},
            label2id={label: i for i, label in enumerate(categories)}
        )
    elif os.path.exists(local_model_path):
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=num_labels)
    else:
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


def setup_model_and_trainer(dataset_dict, le, config, model_name, model, quick=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = len(le.classes_))

    model.config.label2id = {label: i for i, label in enumerate(le.classes_)}
    model.config.id2label = {i: label for i, label in enumerate(le.classes_)}

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=["text", "filename"])

    model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
    os.makedirs(model_save_path, exist_ok=True)

    # Überprüfen, ob es sich um ein ALBERT-Modell handelt
    is_albert = "albert" in model_name.lower()

    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=float(config['Training']['learning_rate']),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1 if quick else int(config['Training']['num_epochs']),
        weight_decay=0.01,
        eval_strategy="steps",  # Geändert von evaluation_strategy zu eval_strategy
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=8,
        logging_dir=os.path.join(model_save_path, 'logs'),
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        gradient_checkpointing=not is_albert,  # Deaktiviere für ALBERT-Modelle
        optim="adamw_torch",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return tokenizer, trainer, tokenized_datasets


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).astype(np.float32).mean().item()}


def predict_top_n(trainer, tokenizer, text, le, n=None):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = trainer.model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    results = []
    for idx, prob in enumerate(probabilities):
        category = le.inverse_transform([idx])[0]
        results.append((category, prob.item()))

    if n is None:
        n = len(le.classes_)

    return sorted(results, key=lambda x: x[1], reverse=True)[:n]


def analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file):
    top_predictions = predict_top_n(trainer, tokenizer, text, le, n=5)  # Top 5 Vorhersagen

    print(f"")
    print(f"Vorhersagen für    {os.path.basename(file_path)}:")
    for category, prob in top_predictions:
        print(f"{category}: {prob:.4f}")

    # Berechnen Sie hier die Gesamtwahrscheinlichkeiten für übergeordnete Kategorien
    # z.B. LRH, Ghostwriter, etc.

    return {
        "Dateiname": os.path.basename(file_path),
        "Dateigröße": file_size,
        "Datum": datetime.now().strftime("%d-%m-%y %H:%M"),
        "Top_Vorhersagen": dict(top_predictions[:5]),
        "Schlussfolgerung": "Komplexe Schlussfolgerung basierend auf den Top-Vorhersagen"
    }

def get_models_for_task(config, task):
    models = []
    for model_info in csv.reader(config['Models']['model_list'].split('\n')):
        if len(model_info) == 4:  # Sicherstellen, dass die Zeile vollständig ist
            hf_name, short_name, train, check = model_info
            if (task == 'train' and train.lower() == 'true') or (task == 'check' and check.lower() == 'true'):
                models.append((hf_name, short_name))
    return models
