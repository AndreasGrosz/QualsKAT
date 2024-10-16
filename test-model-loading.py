import logging
import transformers
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name):
    try:
        logger.info(f"Versuche, Modell zu laden: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        logger.info(f"Modell erfolgreich geladen: {model_name}")
        print ()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells {model_name}: {str(e)}")
        print ()
        return None, None

# Versuchen Sie, jedes Modell zu laden
models = ["distilbert-base-uncased", "roberta-large", "roberta-base", "albert-base-v2", "microsoft_deberta-base"]

for model_name in models:
    tokenizer, model = load_model(model_name)

print(f"Transformers Version: {transformers.__version__}")
print ()

