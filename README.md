# Urheberrechtszuordnung und QKAT CUDA-Projekt

## Überblick
Dieses Projekt kombiniert Techniken zur Urheberrechtszuordnung mit CUDA-beschleunigten Berechnungen und Modellladevorgängen. Ziel ist es, den Autor eines bestimmten Textes sowohl mit generativen (n-Gramm-Sprachmodell) als auch mit diskriminativen (Sequenzklassifikator) Ansätzen zu ermitteln und dabei CUDA zur Leistungsoptimierung zu nutzen.

## Systemanforderungen

- **Betriebssystem**: Kubuntu 24.04 LTS
- **Hardware**: NVIDIA-GPU (für CUDA-Unterstützung)
- **CPU-Architektur**: PowerPC-Unterstützung (für bestimmte PowerPC-Umgebungen)

## Projektkomponenten

1. Urheberrechtsvermerk
- **Generativer Klassifikator**:
- Verwendet das LM-Paket von NLTK für n-Gramm-Sprachmodelle
- Experimentiert mit verschiedenen Glättungstechniken und Backoff-Strategien
- Behandelt Wörter, die nicht im Vokabular enthalten sind
- Generiert Stichproben und meldet die Perplexitätswerte für jeden Autor

- **Diskriminierender Klassifikator**:
- Verwendet Huggingface für die Sequenzklassifizierung
- Bereitet Daten vor, erstellt Trainings- und Test-Datenlader
- Trainiert den Klassifikator mit der Huggingface-Trainer-Klasse

### 2. CUDA und Laden des Modells
- Implementiert CUDA-beschleunigte Berechnungen für eine verbesserte Leistung
- Verwaltet das Laden und Verarbeiten des Modells

## Einrichtung der Umgebung

Dieses Projekt verwendet eine lokale Conda-Umgebung zur Verwaltung von Abhängigkeiten und ist für die Verwendung mit der Fish-Shell konfiguriert.

1. Stelle sicher, dass Anaconda oder Miniconda installiert ist.

2. Installiere die Fish-Shell (falls noch nicht installiert):
```
sudo apt-get update
sudo apt-get install fish
```

3. Lege Fish als Standard-Shell fest:
```
chsh -s /usr/bin/fish
```

4. Klone das Repository:
```
git clone [your-repository-url]
cd [your-project-directory]
```

5. Erstelle und aktiviere die lokale Conda-Umgebung:
```
conda create --prefix ./env python=3.10
conda activate ./env
```

6. Installiere die erforderlichen Pakete:
```
conda env update --prefix ./env --file environment.yml --prune
```

7. Überprüfe die Installation:
```
conda list
```

8. Konfiguriere Fish für Conda:
Füge Folgendes zu deiner `~/.config/fish/config.fish` hinzu:
```
# Conda initialization for Fish
eval /home/your-username/miniconda3/bin/conda „shell.fish“ „hook“ $argv | source
```

## CUDA-Einrichtung

1. Überprüfe die CUDA-Installation:
```
nvidia-smi
nvcc --version
```

2. Stelle sicher, dass die CUDA-Pfade in Fish festgelegt sind:
Füge Folgendes zu `~/.config/fish/config.fish` hinzu:
```
set -gx PATH /usr/local/cuda/bin $PATH
set -gx LD_LIBRARY_PATH /usr/local/cuda/lib64 $LD_LIBRARY_PATH
```

3. Installiere CUDA-fähiges PyTorch:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Projekt ausführen

### Komponente zur Urheberzuordnung
```
python main.py --checkthis --quick

```

## Projektstruktur

- `main.py`: Hauptskript für die Zuordnung der Urheberschaft
- `file_utils.py`: Hilfsfunktionen für Dateioperationen
- `model_utils.py`: Funktionen zum Laden und Verarbeiten von Modellen
- `data_processing.py`: Funktionen zur Datenverarbeitung und -vorbereitung
- `analysis_utils.py`: Analyse-Hilfsfunktionen
- `env/`: Lokale Conda-Umgebung (nicht direkt bearbeiten)
- `models/`: Verzeichnis mit vorab trainierten Modellen
- `CheckThis/`: Verzeichnis für zu analysierende Dateien
- `output/`: Verzeichnis für Ausgabedateien

## Datenvorbereitung
Stelle sicher, dass Quelldateien mit Auszügen verschiedener Autoren im Repository verfügbar sind. Verwende aus Gründen der Konsistenz die UTF-8-Kodierung.
## Datenvorbereitung

[Der vorherige Inhalt bleibt unverändert]

## Git und Umgebungsverwaltung

- Die lokale conda-Umgebung (`./env`) ist in der Versionskontrolle enthalten, mit Ausnahme großer Binärdateien.
- Aktualisiere nach Änderungen an der Umgebung die Datei `environment.yml`:
```
conda env export --from-history > environment.yml
```
- Stelle bei PowerPC-spezifischen Abhängigkeiten sicher, dass sie in der Datei `environment.yml` deutlich gekennzeichnet sind.

## Fehlerbehebung

Bei CUDA-bezogenen Problemen:
1. Stelle sicher, dass deine CUDA-Treiber auf dem neuesten Stand sind.
2. Überprüfe, ob die installierte PyTorch-Version mit deiner CUDA-Version kompatibel ist.
3. Überprüfe das Verzeichnis „models“, um sicherzustellen, dass alle erforderlichen Modelldateien vorhanden sind.
4. Bei PowerPC-spezifischen Problemen konsultiere die Dokumentation zur PowerPC-Architektur und stelle sicher, dass alle Abhängigkeiten kompatibel sind.

## Fish Shell Tipps

- Um die conda-Umgebung in Fish zu aktivieren:
```
conda activate ./env
```
- Um in Fish eine benutzerdefinierte Eingabeaufforderung einzurichten, die die conda-Umgebung anzeigt:
Füge Folgendes zu `~/.config/fish/config.fish` hinzu:
```
function fish_prompt
set -l conda_env (basename „$CONDA_DEFAULT_ENV“)
echo -n „($conda_env) “
set_color $fish_color_cwd
echo -n (prompt_pwd)
set_color normal
echo -n ' > '
end
```

## Mitwirken

[Füge hier alle Richtlinien für die Mitwirkung an deinem Projekt hinzu]

## Lizenz

[Gib die Lizenz an, unter der dein Projekt veröffentlicht wird]

## Umgebung aktivieren
```
conda activate ./env
```
