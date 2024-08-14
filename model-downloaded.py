from transformers import AlbertTokenizer, AlbertForSequenceClassification

model = AlbertForSequenceClassification.from_pretrained("albert-base-v2")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

model.save_pretrained("fresh-models/albert-base-v2")
tokenizer.save_pretrained("fresh-models/albert-base-v2")
