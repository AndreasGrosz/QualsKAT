# Dokumentation für das Skript

## Übersicht

Dieses Skript dient der Verwaltung und Analyse von Dokumenten unter Verwendung von maschinellen Lernmodellen. Es hat zwei Hauptanwendungen:
1. **Training (`--train`)**: Training eines Modells mit den vorhandenen Dokumenten.
2. **Analyse (`--checkthis`)**: Analyse von Dokumenten unter Verwendung eines trainierten Modells.

## Hauptanwendungen

### 1. Training (`--train`)

Das Training des Modells erfolgt, wenn der Parameter `--train` verwendet wird. Die Schritte umfassen:

1. **Initialisierung und Konfiguration**:
   - Das Skript liest die Konfigurationsdatei ein und überprüft die Umgebung.
   - Es wird sichergestellt, dass alle notwendigen Dateien und Verzeichnisse vorhanden sind.

2. **Erstellung des Datensatzes**:
   - Die Funktion `create_dataset(config, quick=False)` aus `data_processing.py` wird verwendet, um den Datensatz aus den vorhandenen Dokumenten zu erstellen.
   - Die Texte werden extrahiert und die Kategorien werden kodiert.

3. **Einrichtung des Modells und des Trainers**:
   - Die Funktion `setup_model_and_trainer(dataset_dict, le, config, model_name, quick=False)` aus `model_utils.py` wird verwendet, um das Modell und den Trainer einzurichten.
   - Das Tokenizer- und Modellobjekt werden geladen und die Trainingsparameter werden konfiguriert.

4. **Training des Modells**:
   - Das Modell wird mit dem erstellten Datensatz trainiert.
   - Nach dem Training wird das Modell gespeichert und die Ergebnisse des Trainings werden protokolliert.

### 2. Analyse (`--checkthis`)

Die Analyse von Dokumenten erfolgt, wenn der Parameter `--checkthis` verwendet wird. Die Schritte umfassen:

1. **Initialisierung und Konfiguration**:
   - Das Skript liest die Konfigurationsdatei ein und überprüft die Umgebung.
   - Es wird sichergestellt, dass alle notwendigen Dateien und Verzeichnisse vorhanden sind.

2. **Laden des Modells**:
   - Das trainierte Modell und der Tokenizer werden geladen.

3. **Erstellung des LabelEncoders**:
   - Ein LabelEncoder wird mit den geladenen Kategorien erstellt, um die Vorhersagen des Modells in menschlich lesbare Kategorien zu übersetzen.

4. **Analyse der Dokumente**:
   - Die Funktion `check_files(trainer, tokenizer, le, config, model_name)` aus `file_utils.py` wird verwendet, um die Dokumente im "CheckThis" Verzeichnis zu analysieren.
   - Der Text wird extrahiert und Vorhersagen werden für jedes Dokument gemacht.
   - Die Ergebnisse werden protokolliert und in einer CSV-Datei gespeichert.

## Detaillierte Beschreibung der Funktionen

### main.py

#### main()

**Beschreibung**:
`main()` ist die zentrale Funktion des Skripts. Sie:

- Initialisiert das Programm und die notwendigen Bibliotheken.
- Liest die Konfigurationsdatei ein und überprüft die Umgebung.
- Trainiert das Modell, wenn der entsprechende Parameter gesetzt ist.
- Führt die Analyse von Dokumenten durch, wenn der entsprechende Parameter gesetzt ist.
- Speichert die Ergebnisse und gibt sie aus.

### analysis_utils.py

#### analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file)

**Beschreibung**:
Die Funktion `analyze_new_article` analysiert einen neuen Artikel. Sie:

- Extrahiert den Text aus einer Datei.
- Führt Vorhersagen für den Text durch.
- Berechnet Wahrscheinlichkeiten und gibt die besten Vorhersagen aus.
- Druckt die Vorhersagen und die berechneten Wahrscheinlichkeiten aus.

### data_processing.py

#### extract_categories(config)

**Beschreibung**:
Extrahiert die Kategorien aus den Dokumentenverzeichnissen und gibt eine Liste der Kategorien zurück.

#### save_categories_to_csv(categories, config)

**Beschreibung**:
Speichert die Kategorien in einer CSV-Datei.

#### get_files_and_categories(config)

**Beschreibung**:
Liest die Dateien und ihre Kategorien aus einem Verzeichnis.

#### load_categories_from_csv(config)

**Beschreibung**:
Lädt Kategorien aus einer CSV-Datei.

#### create_dataset(config, quick=False)

**Beschreibung**:
Erstellt einen Datensatz aus den vorhandenen Dokumenten.

#### extract_text_from_file(file_path)

**Beschreibung**:
Extrahiert den Text aus einer Datei basierend auf ihrem Format.

#### extract_text_from_old_doc(file_path)

**Beschreibung**:
Extrahiert den Text aus alten DOC-Dateien.

#### handle_rtf_error(file_path)

**Beschreibung**:
Handhabt das Lesen von RTF-Dateien und konvertiert sie in reinen Text.

### model_utils.py

#### get_model_and_tokenizer(model_name, num_labels)

**Beschreibung**:
Lädt das Tokenizer- und Modellobjekt für ein bestimmtes Modell.

#### setup_model_and_trainer(dataset_dict, le, config, model_name, quick=False)

**Beschreibung**:
Richtet das Modell und den Trainer ein.

#### compute_metrics(eval_pred)

**Beschreibung**:
Berechnet die Genauigkeit der Vorhersagen.

#### predict_top_n(trainer, tokenizer, text, le, n=None)

**Beschreibung**:
Führt Vorhersagen für den gegebenen Text durch und gibt die Top-N Ergebnisse zurück.

#### analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file)

**Beschreibung**:
Analysiert einen neuen Artikel, ähnlich wie in `analysis_utils.py`.

### file_utils.py

#### check_environment()

**Beschreibung**:
Überprüft die Umgebung, um sicherzustellen, dass alle notwendigen Dateien und Verzeichnisse vorhanden sind und die GPU-Verfügbarkeit.

#### check_hf_token()

**Beschreibung**:
Überprüft, ob der Hugging Face API-Token gültig ist.

#### check_files(trainer, tokenizer, le, config, model_name)

**Beschreibung**:
Überprüft und analysiert Dateien im "CheckThis" Verzeichnis.

#### extract_text_from_file(file_path)

**Beschreibung**:
Extrahiert den Text aus einer Datei basierend auf ihrem Format, ähnlich wie in `data_processing.py`.

#### extract_text_from_old_doc(file_path)

**Beschreibung**:
Extrahiert den Text aus alten DOC-Dateien, ähnlich wie in `data_processing.py`.

#### handle_rtf_error(file_path)

**Beschreibung**:
Handhabt das Lesen von RTF-Dateien und konvertiert sie in reinen Text, ähnlich wie in `data_processing.py`.

---

Diese Dokumentation bietet eine strukturierte Übersicht über das Skript und seine Hauptanwendungen. Sie erklärt, wie das Skript verwendet wird, um Modelle zu trainieren und Dokumente zu analysieren, und gibt detaillierte Beschreibungen der wichtigsten Funktionen in den einzelnen Dateien.



# main.py

## main()

**Beschreibung**:
`main()` ist die zentrale Funktion des Skripts. Sie:

- Initialisiert das Programm und die notwendigen Bibliotheken.
- Liest die Konfigurationsdatei ein und überprüft die Umgebung.
- Trainiert das Modell, wenn der entsprechende Parameter gesetzt ist.
- Führt die Analyse von Dokumenten durch, wenn der entsprechende Parameter gesetzt ist.
- Speichert die Ergebnisse und gibt sie aus.

Für einen AI-Programmierlaien:
Dieses Skript ist das "Hauptbuch", das bestimmt, was das Programm tun soll. Es liest die Einstellungen, trainiert das Modell oder analysiert Dokumente basierend auf den Eingaben des Benutzers.

### 2. `analysis_utils.py`

# analysis_utils.py

## analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file)

**Beschreibung**:
Die Funktion `analyze_new_article` analysiert einen neuen Artikel. Sie:

- Extrahiert den Text aus einer Datei.
- Führt Vorhersagen für den Text durch.
- Berechnet Wahrscheinlichkeiten und gibt die besten Vorhersagen aus.
- Druckt die Vorhersagen und die berechneten Wahrscheinlichkeiten aus.

Für einen AI-Programmierlaien:
Diese Funktion liest den Text eines Artikels, führt eine Analyse durch, und sagt, zu welcher Kategorie der Artikel am wahrscheinlichsten gehört.

### 3. `data_processing.py`

# data_processing.py

## extract_categories(config)

**Beschreibung**:
Extrahiert die Kategorien aus den Dokumentenverzeichnissen und gibt eine Liste der Kategorien zurück.

Für einen AI-Programmierlaien:
Diese Funktion durchsucht die Verzeichnisse nach Dokumentenkategorien.

## save_categories_to_csv(categories, config)

**Beschreibung**:
Speichert die Kategorien in einer CSV-Datei.

Für einen AI-Programmierlaien:
Diese Funktion speichert die gefundenen Kategorien in einer Datei zur späteren Verwendung.

## get_files_and_categories(config)

**Beschreibung**:
Liest die Dateien und ihre Kategorien aus einem Verzeichnis.

Für einen AI-Programmierlaien:
Diese Funktion sammelt alle Dokumente und ihre zugehörigen Kategorien.

## load_categories_from_csv(config)

**Beschreibung**:
Lädt Kategorien aus einer CSV-Datei.

Für einen AI-Programmierlaien:
Diese Funktion liest die Kategorien aus einer zuvor gespeicherten Datei.

## create_dataset(config, quick=False)

**Beschreibung**:
Erstellt einen Datensatz aus den vorhandenen Dokumenten.

Für einen AI-Programmierlaien:
Diese Funktion sammelt alle Texte und Kategorien aus den Dokumenten, um sie in eine Form zu bringen, die das Modell verstehen und damit arbeiten kann.

## extract_text_from_file(file_path)

**Beschreibung**:
Extrahiert den Text aus einer Datei basierend auf ihrem Format.

Für einen AI-Programmierlaien:
Diese Funktion liest den Text aus verschiedenen Dokumenttypen (z.B. TXT, RTF, DOCX) aus.

## extract_text_from_old_doc(file_path)

**Beschreibung**:
Extrahiert den Text aus alten DOC-Dateien.

Für einen AI-Programmierlaien:
Diese Funktion liest den Text aus alten Microsoft Word-Dokumenten aus.

## handle_rtf_error(file_path)

**Beschreibung**:
Handhabt das Lesen von RTF-Dateien und konvertiert sie in reinen Text.

Für einen AI-Programmierlaien:
Diese Funktion liest den Text aus RTF-Dokumenten aus.

### 4. `model_utils.py`

# model_utils.py

## get_model_and_tokenizer(model_name, num_labels)

**Beschreibung**:
Lädt das Tokenizer- und Modellobjekt für ein bestimmtes Modell.

Für einen AI-Programmierlaien:
Diese Funktion lädt das AI-Modell und das Werkzeug, das Texte in eine für das Modell verständliche Form bringt.

## setup_model_and_trainer(dataset_dict, le, config, model_name, quick=False)

**Beschreibung**:
Richtet das Modell und den Trainer ein.

Für einen AI-Programmierlaien:
Diese Funktion bereitet das AI-Modell vor, indem sie es einrichtet und ihm sagt, wie es lernen und trainieren soll.

## compute_metrics(eval_pred)

**Beschreibung**:
Berechnet die Genauigkeit der Vorhersagen.

Für einen AI-Programmierlaien:
Diese Funktion misst, wie genau das Modell Vorhersagen trifft.

## predict_top_n(trainer, tokenizer, text, le, n=None)

**Beschreibung**:
Führt Vorhersagen für den gegebenen Text durch und gibt die Top-N Ergebnisse zurück.

Für einen AI-Programmierlaien:
Diese Funktion analysiert den Text und gibt die wahrscheinlichsten Kategorien zurück.

## analyze_new_article(file_path, trainer, tokenizer, le, extract_text_from_file)

**Beschreibung**:
Analysiert einen neuen Artikel, ähnlich wie in `analysis_utils.py`.

Für einen AI-Programmierlaien:
Diese Funktion liest den Text eines Artikels, führt eine Analyse durch und gibt die wahrscheinlichsten Kategorien zurück.

### 5. `file_utils.py`

# file_utils.py

## check_environment()

**Beschreibung**:
Überprüft die Umgebung, um sicherzustellen, dass alle notwendigen Dateien und Verzeichnisse vorhanden sind und die GPU-Verfügbarkeit.

Für einen AI-Programmierlaien:
Diese Funktion stellt sicher, dass alles bereit ist und funktioniert, bevor das eigentliche Programm ausgeführt wird.

## check_hf_token()

**Beschreibung**:
Überprüft, ob der Hugging Face API-Token gültig ist.

Für einen AI-Programmierlaien:
Diese Funktion stellt sicher, dass der Zugang zu den benötigten Online-Ressourcen funktioniert.

## check_files(trainer, tokenizer, le, config, model_name)

**Beschreibung**:
Überprüft und analysiert Dateien im "CheckThis" Verzeichnis.

Für einen AI-Programmierlaien:
Diese Funktion analysiert Dokumente und gibt die Ergebnisse aus.

## extract_text_from_file(file_path)

**Beschreibung**:
Extrahiert den Text aus einer Datei basierend auf ihrem Format, ähnlich wie in `data_processing.py`.

Für einen AI-Programmierlaien:
Diese Funktion liest den Text aus verschiedenen Dokumenttypen (z.B. TXT, RTF, DOCX, PDF) aus.

## extract_text_from_old_doc(file_path)

**Beschreibung**:
Extrahiert den Text aus alten DOC-Dateien, ähnlich wie in `data_processing.py`.

Für einen AI-Programmierlaien:
Diese Funktion liest den Text aus alten Microsoft Word-Dokumenten aus.

## handle_rtf_error(file_path)

**Beschreibung**:
Handhabt das Lesen von RTF-Dateien und konvertiert sie in reinen Text, ähnlich wie in `data_processing.py`.

Für einen AI-Programmierlaien:
Diese Funktion liest den Text aus RTF-Dokumenten aus.

## Flussdiagramm

```mermaid
graph TD
    A[Start] --> B[main()]
    B --> C[Initialisierung und Überprüfung der Umgebung]
    C --> D{--train?}
    D -->|Ja| E[create_dataset()]
    D -->|Nein| F
    E --> G[setup_model_and_trainer()]
    G --> H[train()]
    H --> I[evaluate()]
    I --> J[save_model()]
    J --> K[End]
    F --> L{--checkthis?}
    L -->|Ja| M[check_files()]
    L -->|Nein| N
    M --> O[analyze_new_article()]
    O --> P[extract_text_from_file()]
    P --> Q[predict_top_n()]
    Q --> R[Ergebnis speichern]
    R --> K
    N --> K




### Erklärung des Flussdiagramms

- **Start**: Beginn des Skripts.
- **main()**: Zentrale Funktion, die das Skript ausführt.
- **Initialisierung und Überprüfung der Umgebung**: Überprüfung der Konfiguration und Umgebung, um sicherzustellen, dass alle notwendigen Dateien und Verzeichnisse vorhanden sind.
- **--train**: Wenn der `--train` Parameter gesetzt ist, wird das Modell trainiert.
  - **create_dataset()**: Erstellen des Datensatzes aus den vorhandenen Dokumenten.
  - **setup_model_and_trainer()**: Einrichten des Modells und des Trainers.
  - **train()**: Training des Modells.
  - **evaluate()**: Bewertung des Modells.
  - **save_model()**: Speichern des trainierten Modells.
- **--checkthis**: Wenn der `--checkthis` Parameter gesetzt ist, werden die Dokumente analysiert.
  - **check_files()**: Überprüfen und Analysieren der Dateien im "CheckThis" Verzeichnis.
  - **analyze_new_article()**: Analysieren eines neuen Artikels.
  - **extract_text_from_file()**: Extrahieren des Textes aus einer Datei.
  - **predict_top_n()**: Vorhersagen für den gegebenen Text durchführen.
  - **Ergebnis speichern**: Speichern der Analyseergebnisse.
- **End**: Ende des Skripts.

Dieses Flussdiagramm gibt einen visuellen Überblick über die Hauptfunktionen und ihre Beziehungen im Skript, basierend auf den beiden Hauptanwendungen `--train` und `--checkthis`.
