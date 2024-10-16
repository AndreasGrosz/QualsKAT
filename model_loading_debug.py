import logging
import os
import json
from transformers import AutoConfig, AutoTokenizer, AutoModel
import transformers
from transformers import AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_MODEL_PATH = "/home/res/projekte/kd0241-py/Q-KAT/AAUNaL/Beta-v1.2/models-240815"

def check_file_existence(path):
    logger.debug(f"Überprüfe Existenz von: {path}")
    exists = os.path.exists(path)
    logger.debug(f"Datei existiert: {exists}")
    return exists

def load_model(model_name):
    try:
        logger.info(f"Versuche, Modell zu laden: {model_name}")
        model_path = os.path.join(BASE_MODEL_PATH, model_name)

        if not os.path.exists(model_path):
            raise ValueError(f"Modellverzeichnis nicht gefunden: {model_path}")

        logger.debug(f"Inhalt des Modellverzeichnisses: {os.listdir(model_path)}")

        config_path = os.path.join(model_path, "config.json")
        if not check_file_existence(config_path):
            raise ValueError(f"config.json nicht gefunden in: {model_path}")

        with open(config_path, 'r') as f:
            config_data = json.load(f)
            logger.debug(f"Inhalt von config.json: {json.dumps(config_data, indent=2)}")

        logger.debug("Versuche, Konfiguration zu laden...")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        logger.debug(f"Konfiguration geladen: {config}")

        logger.debug("Versuche, Tokenizer zu laden...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        logger.debug("Tokenizer erfolgreich geladen")

        logger.debug("Versuche, Modell zu laden...")
        model = AutoModel.from_pretrained(model_path, config=config, local_files_only=True)
        logger.debug("Modell erfolgreich geladen")

        logger.info(f"Modell erfolgreich geladen: {model_name}")
        logger.info(f"Modell Konfiguration für {model_name}:")
        logger.info(f"Label2id: {model.config.label2id}")
        logger.info(f"Id2label: {model.config.id2label}")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells {model_name}: {str(e)}", exc_info=True)
        return None, None

# Liste der Modellnamen
models = ["distilbert-base-uncased", "roberta-large", "roberta-base", "albert-base-v2", "microsoft_deberta-base"]

# Versuchen Sie, jedes Modell zu laden
for model_name in models:
    tokenizer, model = load_model(model_name)

logger.info(f"Transformers Version: {transformers.__version__}")

def check_label_encoder():
    categories = ['Non-LRH', 'LRH']
    le = LabelEncoder()
    le.fit(categories)
    logger.info(f"Label-Encoder Klassen: {le.classes_}")
    logger.info(f"Label-Encoder Transformationen: LRH -> {le.transform(['LRH'])}, Non-LRH -> {le.transform(['Non-LRH'])}")

if __name__ == "__main__":
    # Versuchen Sie, jedes Modell zu laden
    for model_name in models:
        tokenizer, model = load_model(model_name)

    logger.info(f"Transformers Version: {transformers.__version__}")

    # Überprüfen Sie den Label-Encoder
    check_label_encoder()
