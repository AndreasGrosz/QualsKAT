import nltk
import os
import torch
import numpy as np
import pandas as pd
import random
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize, word_tokenize
import argparse

# Set NLTK data path
nltk.data.path.append("./nltk_data")

# Check for the Hugging Face token
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if hf_token is None:
    raise ValueError("Hugging Face API token is missing. Please set the environment variable 'HUGGINGFACE_HUB_TOKEN'.")

# Validate the token
api = HfApi()
try:
    api.whoami(token=hf_token)
except HTTPError as e:
    raise ValueError("Invalid Hugging Face API token. Please check the token and try again.")

def get_files_and_categories(root_dir, test_mode=False):
    files_and_categories = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                # Extract categories from the file path
                categories = os.path.relpath(root, root_dir).split(os.sep)
                files_and_categories.append((file_path, categories))

                if test_mode and len(files_and_categories) >= 10:
                    return files_and_categories
    return files_and_categories

def create_dataset(root_dir, test_mode=False):
    files_and_categories = get_files_and_categories(root_dir, test_mode)
    texts = []
    all_categories = []

    for file_path, categories in files_and_categories:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        if test_mode:
            text = text[:1000]  # Limit to first 1000 characters in test mode

        texts.append(text)
        all_categories.append("-".join(categories))  # Join categories into a single string

    # Create a LabelEncoder to transform categories into numeric labels
    le = LabelEncoder()
    numeric_categories = le.fit_transform(all_categories)

    # Create dataset dictionary
    dataset_dict = {
        'text': texts,
        'labels': numeric_categories
    }

    return Dataset.from_dict(dataset_dict), le

def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(examples["text"], truncation=True)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def main(args):
    root_dir = 'documents'
    test_mode = args.test_mode

    dataset, le = create_dataset(root_dir, test_mode)

    # Split dataset into train, validation, and test
    dataset = dataset.train_test_split(test_size=0.2)
    test_valid = dataset['test'].train_test_split(test_size=0.5)
    dataset_dict = DatasetDict({
        'train': dataset['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']
    })

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_dataset = dataset_dict.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)

    num_labels = len(le.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir='Multi_Class_Classifier',
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
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate on test set
    results = trainer.evaluate(tokenized_dataset['test'])
    print(results)

    # Example of prediction
    text = "This is a sample text to classify."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_category = le.inverse_transform([predicted_class])[0]
    print(f"Predicted category: {predicted_category}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Class Text Classification')
    parser.add_argument('-approach', choices=['discriminative'], required=True)
    parser.add_argument('-test_mode', action='store_true', help='Run in test mode with limited data')
    args = parser.parse_args()
    main(args)
