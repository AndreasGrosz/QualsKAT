from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

def setup_model_and_trainer(dataset_dict, le, config, model_name, quick=False):
    num_labels = len(le.classes_)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    model.config.label2id = {label: i for i, label in enumerate(le.classes_)}
    model.config.id2label = {i: label for i, label in enumerate(le.classes_)}

    def tokenize_function(examples):
        result = tokenizer(examples["text"], truncation=True, padding="max_length")
        result["labels"] = examples["labels"]
        return result

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

# Rest der Datei bleibt unverändert
