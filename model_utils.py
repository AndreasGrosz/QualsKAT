import csv
from datetime import datetime
import warnings
from data_processing import extract_text_from_file
import datetime
import torch
from safetensors.torch import save_file
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, LlamaForCausalLM, LlamaTokenizer, AutoConfig,get_linear_schedule_with_warmup, TrainerCallback
import numpy as np
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm


warnings.filterwarnings("ignore", message="Some weights of")


def get_optimizer_and_scheduler(model, config, num_training_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": float(config['Training']['weight_decay']),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(config['Training']['learning_rate']))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['Training']['warmup_steps']),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


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

    if os.path.exists(local_model_path):
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)

        # Lade die Konfiguration und passe sie an
        model_config = AutoConfig.from_pretrained(local_model_path)
        model_config.num_labels = num_labels
        model_config.id2label = {i: label for i, label in enumerate(categories)}
        model_config.label2id = {label: i for i, label in enumerate(categories)}

        # Initialisiere das Modell mit der angepassten Konfiguration
        model = AutoModelForSequenceClassification.from_pretrained(
            local_model_path,
            config=model_config,
            ignore_mismatched_sizes=True
        )
        if "xlnet" in model_name.lower():
            model.train()  # Stellen Sie sicher, dass das Modell im Trainingsmodus ist
            # Initialisieren Sie die Speicher für die relativen Positionen
            model.transformer.mem_len = 2048
            model.transformer.attn_type = 'bi'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={i: label for i, label in enumerate(categories)},
            label2id={label: i for i, label in enumerate(categories)},
            ignore_mismatched_sizes=True
        )

    return model, tokenizer


def make_model_tensors_contiguous(model):
    for param in model.state_dict().values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.contiguous()


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


def setup_model_and_trainer(dataset, le, config, model_name, model, tokenizer, quick=False):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    # Tokenisiere den gesamten Datensatz
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text", "filename"])

    # Teile den Datensatz in Train und Test
    train_testvalid = tokenized_datasets.train_test_split(test_size=0.3, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

    tokenized_datasets = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']
    })

    model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
    os.makedirs(model_save_path, exist_ok=True)

    # Überprüfen, ob es sich um ein ALBERT- oder XLNet-Modell handelt
    is_albert = "albert" in model_name.lower()
    is_xlnet = "xlnet" in model_name.lower()

    # Batch-Größe für XLNet reduzieren
    batch_size = 1 if is_xlnet else int(config['Training']['batch_size'])

    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=float(config['Training']['learning_rate']),
        per_device_train_batch_size=int(config['Training']['batch_size']),
        per_device_eval_batch_size=int(config['Training']['batch_size']),
        num_train_epochs=1 if quick else int(config['Training']['num_epochs']),
        weight_decay=float(config['Training']['weight_decay']),
        evaluation_strategy="steps",
        eval_steps=int(config['Evaluation']['eval_steps']),
        save_strategy="steps",
        save_steps=int(config['Evaluation']['save_steps']),
        load_best_model_at_end=True,
        fp16=config['Optimization'].getboolean('fp16') and torch.cuda.is_available(),
        gradient_accumulation_steps=int(config['Training']['gradient_accumulation_steps']),
        logging_dir=os.path.join(model_save_path, 'logs'),
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    if "t5" in model_name.lower():
        training_args.gradient_checkpointing = True
    elif "xlnet" in model_name.lower() or "albert" in model_name.lower():
        training_args.gradient_checkpointing = False
    else:
        training_args.gradient_checkpointing = True
    if "xlnet" in model_name.lower():
        training_args.per_device_train_batch_size = int(config['Training']['xlnet_batch_size'])
        training_args.gradient_accumulation_steps = int(config['Training']['xlnet_gradient_accumulation_steps'])
    # Batch-Größe für große Modelle reduzieren
    if any(model in model_name.lower() for model in ['large', 't5', 'xlnet', 'longformer']):
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        training_args.gradient_accumulation_steps = 16
    num_update_steps_per_epoch = len(tokenized_datasets['train']) // (int(config['Training']['batch_size']) * int(config['Training']['gradient_accumulation_steps']))
    num_train_epochs = 1 if quick else int(config['Training']['num_epochs'])
    total_steps = num_update_steps_per_epoch * num_train_epochs

    training_args = TrainingArguments(
        # ... (bestehende Argumente)
        max_grad_norm=float(config['Optimization']['max_grad_norm']),
    )

    optimizer, scheduler = get_optimizer_and_scheduler(model, config, total_steps)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    optimizer, scheduler = get_optimizer_and_scheduler(model, config, total_steps)

    def setup_model_and_trainer(dataset, le, config, model_name, model, tokenizer, quick=False):

    class SafetensorsSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            model_to_save = kwargs['model'].module if hasattr(kwargs['model'], 'module') else kwargs['model']
            save_file(model_to_save.state_dict(), os.path.join(checkpoint_folder, 'model.safetensors'))
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        callbacks=[SafetensorsSaveCallback]
    )

    return trainer, tokenized_datasets


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
