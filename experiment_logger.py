import logging
import json
import os
from datetime import datetime

def log_experiment(config, model_name, results, output_dir):
    experiment_log = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "hyperparameters": {
            "learning_rate": config['Training']['learning_rate'],
            "batch_size": config['Training']['batch_size'],
            "num_epochs": config['Training']['num_epochs']
        },
        "results": results
    }

    # Erstelle das Ausgabeverzeichnis, falls es nicht existiert
    os.makedirs(output_dir, exist_ok=True)

    # Bereinige den Modellnamen für die Verwendung im Dateinamen
    safe_model_name = model_name.replace('/', '_')
    log_file = os.path.join(output_dir, f"experiment_log_{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    try:
        with open(log_file, 'w') as f:
            json.dump(experiment_log, f, indent=2)
        logging.info(f"Experiment log saved to {log_file}")
    except Exception as e:
        logging.error(f"Failed to save experiment log: {str(e)}")

