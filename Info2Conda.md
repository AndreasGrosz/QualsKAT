# Grundinfos zur Conda-Umgebung

conda speichert seine Umgebungen standardmäßig nicht im lokalen Projektverzeichnis.

## Warum von venv zu Conda wechseln:

Es gibt gute Gründe, warum man bei CUDA-Projekten von venv zu conda wechseln könnte:

1.  CUDA-Kompatibilität:
    Conda bietet bessere Unterstützung für CUDA-Pakete und deren Abhängigkeiten. Es kann CUDA-Treiber und cuDNN-Bibliotheken direkt verwalten, was mit venv oft schwieriger ist.
2.  Konsistente Versionen:
    Conda kann besser sicherstellen, dass alle CUDA-bezogenen Pakete (PyTorch, cuDNN, etc.) in kompatiblen Versionen installiert werden.
3.  Einfachere GPU-Paket-Installation:
    Mit conda ist es oft einfacher, GPU-unterstützte Versionen von Paketen wie PyTorch zu installieren, da conda automatisch die richtigen CUDA-Abhängigkeiten auflöst.
4.  Binäre Pakete:
    Conda bietet oft vorcompilierte Binärpakete, was die Installation von komplexen Paketen mit CUDA-Unterstützung erleichtern kann.
5.  Umgebungsmanagement:
    Conda bietet robustere Werkzeuge zum Verwalten von Umgebungen, einschließlich der Möglichkeit, nicht-Python-Abhängigkeiten zu handhaben.
6.  Reproduzierbarkeit:
    Mit conda-forge und anderen Kanälen kann man spezifische Versionen von Paketen und deren Abhängigkeiten genauer festlegen.
7.  Systemweite vs. Benutzer-Installation:
    Conda ermöglicht eine sauberere Trennung zwischen systemweiten und benutzerspezifischen Installationen.
8.  Multi-Sprach-Support:
    Falls Ihr Projekt neben Python auch andere Sprachen oder Tools verwendet, kann conda diese ebenfalls verwalten.

Trotz dieser Vorteile ist es wichtig zu beachten, dass venv für viele Python-Projekte ausreichend ist und manchmal einfacher zu handhaben sein kann, besonders für reine Python-Projekte ohne komplexe Abhängigkeiten.


## Conda-Umgebungen:
Conda speichert Umgebungen standardmäßig in einem zentralen Verzeichnis, üblicherweise unter:

~/anaconda3/envs/ oder
~/miniconda3/envs/

## Überprüfen der aktuellen Conda-Umgebungen:

conda env list

Dies zeigt Ihnen alle verfügbaren Conda-Umgebungen.

## Aktivieren der richtigen Umgebung:
Wenn Sie eine Umgebung namens "aaunal" oder ähnlich sehen, aktivieren Sie diese:

conda activate aaunal

## Überprüfen der aktuellen Python-Version und installierten Pakete:

python --version
conda list

## Wenn keine passende Umgebung vorhanden ist, erstellen Sie eine neue:

conda create -n aaunal python=3.10
conda activate aaunal

## Installation der erforderlichen Pakete:

Wenn Sie eine environment.yml Datei haben:

conda env update -f environment.yml

Oder wenn Sie eine requirements.txt Datei haben:
Copypip install -r requirements.txt

## Überprüfen Sie die CUDA-Verfügbarkeit in der Conda-Umgebung:

import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

## Versuchen Sie, ein Modell in der Conda-Umgebung zu laden:

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

model_name = "distilbert-base-uncased"
model_path = os.path.join('models', model_name)
print(f"Versuche, Modell zu laden von: {model_path}")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Modell erfolgreich geladen!")

## Umgebung deactivieren

conda deactivate

## Hauptskript ausführen:

main.py --checkthis --quick

## Conda lokal:

Für zukünftige Projekte könnten Sie in Erwägung ziehen, projektspezifische Conda-Umgebungen zu verwenden:

conda create -p ./env python=3.10
conda activate ./env

Dies erstellt eine Umgebung im aktuellen Projektverzeichnis.
