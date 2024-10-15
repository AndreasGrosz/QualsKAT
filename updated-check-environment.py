from transformers import AutoModel, AutoTokenizer
import os

def check_environment():
       config = {}
       model_name = "distilbert-base-uncased"
       model_path = os.path.join("fresh-models", model_name)

       try:
           if not os.path.exists(model_path):
               print(f"Modell nicht gefunden. Lade {model_name} herunter...")
               tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
               model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
               print("Modell erfolgreich heruntergeladen und gespeichert.")
           else:
               print(f"Lade vorhandenes Modell aus {model_path}")
               tokenizer = AutoTokenizer.from_pretrained(model_path)
               model = AutoModel.from_pretrained(model_path)

           config['model_name'] = model_name
           config['model_path'] = model_path
       except Exception as e:
           raise ValueError(f"Fehler beim Laden/Herunterladen des Modells {model_name}: {str(e)}")

       # Weitere Umgebungsprüfungen hier...

       return config
   
