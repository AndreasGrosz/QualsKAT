from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_names = ["distilbert-base-uncased", "albert-base-v2", "roberta-large", "roberta-base", "microsoft/deberta-base"]

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer.save_pretrained(f"fresh-models/{model_name}")
    model.save_pretrained(f"fresh-models/{model_name}")
