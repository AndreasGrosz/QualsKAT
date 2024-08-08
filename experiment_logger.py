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

    log_file = os.path.join(output_dir, f"experiment_log_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=2)

    print(f"Experiment log saved to {log_file}")

# Verwendung in Ihrem Hauptskript:
# log_experiment(config, model_name, results, config['Paths']['output'])
