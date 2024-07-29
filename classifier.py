import os
import nltk
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize, word_tokenize
import argparse
import docx
from pypdf import PdfReader
import pptx
import xlrd
import logging
import time
from bs4 import BeautifulSoup
import striprtf.striprtf as striprtf



# Konfiguriere Logging
logging.basicConfig(filename='file_errors.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Set NLTK data path
nltk.data.path.append("./nltk_data")

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
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252', 'iso-8859-15', 'mac_roman']

    if file_path.endswith('.txt'):
        # Spezifische Behandlung für .txt Dateien
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError as e:
                logging.error(f"UnicodeDecodeError for file {file_path} with encoding {encoding}: {e}")
        logging.error(f"Unable to read the .txt file {file_path} with supported encodings.")
        return None
    elif file_path.endswith('.docx'):
        try:
            doc = docx.Document(file_path)
            return ' '.join([para.text for para in doc.paragraphs])
        except Exception as e:
            logging.error(f"Error reading .docx file {file_path}: {e}")
            return None
    elif file_path.endswith('.pdf'):
        try:
            reader = PdfReader(file_path)
            return ' '.join([page.extract_text() for page in reader.pages])
        except Exception as e:
            logging.error(f"Error reading .pdf file {file_path}: {e}")
            return None
    elif file_path.endswith('.pptx'):
        try:
            ppt = pptx.Presentation(file_path)
            return ' '.join([shape.text for slide in ppt.slides for shape in slide.shapes if hasattr(shape, 'text')])
        except Exception as e:
            logging.error(f"Error reading .pptx file {file_path}: {e}")
            return None
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        try:
            workbook = xlrd.open_workbook(file_path)
            return ' '.join([sheet.cell_value(row, col)
                             for sheet in workbook.sheets()
                             for row in range(sheet.nrows)
                             for col in range(sheet.ncols)
                             if sheet.cell_value(row, col)])
        except Exception as e:
            logging.error(f"Error reading Excel file {file_path}: {e}")
            return None
    elif file_path.endswith('.rtf'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                rtf_text = file.read()
                return striprtf.striprtf(rtf_text)
        except Exception as e:
            logging.error(f"Error reading .rtf file {file_path}: {e}")
            return None

    elif file_path.endswith('.html') or file_path.endswith('.htm'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                return soup.get_text()
        except Exception as e:
            logging.error(f"Error reading HTML file {file_path}: {e}")
            return None

    else:
        logging.error(f"Unsupported file type: {file_path}")
        return None

def get_files_and_categories(root_dir, test_mode=False):
    files_and_categories = []
    supported_formats = ['.txt', '.doc', '.docx', '.rtf', '.pdf', '.html', '.htm']

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(fmt) for fmt in supported_formats):
                file_path = os.path.join(root, file)
                categories = os.path.relpath(root, root_dir).split(os.sep)
                if 'unknown' in categories:
                    categories = ['unknown']
                files_and_categories.append((file_path, categories))

                if test_mode and len(files_and_categories) >= 10:
                    return files_and_categories
    return files_and_categories

def create_dataset(root_dir, test_mode=False):
    files_and_categories = get_files_and_categories(root_dir, test_mode)
    texts = []
    all_categories = []

    for file_path, categories in files_and_categories:
        text = extract_text_from_file(file_path)
        if text:
            if test_mode:
                text = text[:10000]  # Limit to first 10000 characters in test mode
            texts.append(text)
            all_categories.append(categories[0])  # Nehmen Sie nur die erste Kategorie

    le = LabelEncoder()
    numeric_categories = le.fit_transform(all_categories)

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

def predict_with_confidence(trainer, tokenizer, text, le):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = trainer.model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    predicted_category = le.inverse_transform([predicted_class])[0]
    return predicted_category, confidence

def main(args):
    start_time = time.time()
    print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    check_hf_token()

    root_dir = 'documents'
    test_mode = args.test_mode

    dataset, le = create_dataset(root_dir, test_mode)

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

    results = trainer.evaluate(tokenized_dataset['test'])
    print("Test results:", results)

    # Example predictions
    example_texts = [
        "This is a sample text to classify.",
        "Another example of unknown text.",
        "A text that might be from LRH in the 1950s."
    ]

    confidence_threshold = 0.7  # Set your desired threshold

    for text in example_texts:
        predicted_category, confidence = predict_with_confidence(trainer, tokenizer, text, le)
        if confidence < confidence_threshold:
            print(f"Text: '{text[:50]}...' | Prediction: Possibly Unknown | Confidence: {confidence:.2f}")
        else:
            print(f"Text: '{text[:50]}...' | Prediction: {predicted_category} | Confidence: {confidence:.2f}")
    end_time = time.time()
    print(f"Script ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Format Document Classification')
    parser.add_argument('-approach', choices=['discriminative'], required=True)
    parser.add_argument('-test_mode', action='store_true', help='Run in test mode with limited data')
    args = parser.parse_args()
    main(args)
