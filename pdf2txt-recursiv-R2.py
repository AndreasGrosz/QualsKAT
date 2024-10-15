"""
# Aktualisierte Anleitung: Einrichtung und Aktualisierung des virtuellen Environments

## Vorbereitungen

1. Navigieren Sie in Ihr Projektverzeichnis:
   ```fish
   cd pfad/zu/ihrem/projekt
   ```

2. Aktivieren Sie Ihr virtuelles Environment:
   ```fish
   source .venv/bin/activate.fish
   ```
   Oder, wenn Sie die `venv`-Funktion eingerichtet haben:
   ```fish
   venv
   ```

## Aktualisierung und Installation

1. Aktualisieren Sie pip auf die neueste Version:
   ```fish
   pip install --upgrade pip
   ```

2. Sichern Sie Ihre aktuelle `requirements.txt`:
   ```fish
   cp requirements.txt requirements.txt.bak
   ```

3. Installieren Sie die Pakete aus `requirements.txt`:
   ```fish
   pip install -r requirements.txt
   ```

4. Falls Versionskonflikte auftreten:
   a) Öffnen Sie `requirements.txt` in einem Texteditor.
   b) Suchen Sie nach konfliktverursachenden Paketen (z.B. numpy, accelerate).
   c) Passen Sie die Versionsangaben an, z.B.:
      ```
      numpy>=1.17,<2.0.0
      accelerate>=0.33.0
      ```
   d) Speichern Sie die Änderungen und versuchen Sie die Installation erneut.

5. Bei anhaltenden Problemen:
   a) Installieren Sie kritische Pakete manuell:
      ```fish
      pip install numpy==1.24.3
      pip install -r requirements.txt
      ```
   b) Oder erwägen Sie die Neuerstellung des virtuellen Environments:
      ```fish
      deactivate
      rm -rf .venv
      python3 -m venv .venv
      source .venv/bin/activate.fish
      pip install --upgrade pip
      pip install -r requirements.txt
      ```

6. Nach erfolgreicher Installation, aktualisieren Sie `requirements.txt`:
   ```fish
   pip freeze > requirements.txt
   ```

## Überprüfung

Überprüfen Sie die Installation:
```fish
pip list
```

## Tipps

- Verwenden Sie `pip install -v -r requirements.txt` für detailliertere Ausgaben.
- Bei komplexen Abhängigkeiten, erwägen Sie die Verwendung von `pip-tools` für besseres Dependency-Management.
- Halten Sie Ihr virtuelles Environment und `requirements.txt` regelmäßig aktualisiert, um zukünftige Konflikte zu minimieren.
"""
import os
import sys
from PyPDF2 import PdfReader

def convert_pdf_to_txt(source_dir, target_dir):
    # Erstelle das Zielverzeichnis, falls es nicht existiert
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Durchsuche alle Unterordner im Quellverzeichnis
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                # Erstelle die Pfade für Quell- und Zieldateien
                pdf_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_dir)
                target_subdir = os.path.join(target_dir, relative_path)

                # Erstelle Unterverzeichnisse im Zielordner, falls nötig
                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)

                txt_file_name = os.path.splitext(file)[0] + '.txt'
                txt_file_path = os.path.join(target_subdir, txt_file_name)

                print(f"Konvertiere {pdf_file_path} nach {txt_file_path}...")

                try:
                    # Lese die PDF und extrahiere den Text
                    with open(pdf_file_path, 'rb') as pdf_file:
                        reader = PdfReader(pdf_file)
                        text = ''
                        for page in reader.pages:
                            text += page.extract_text()

                    # Schreibe den extrahierten Text in eine TXT-Datei
                    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(text)

                    print(f"Konvertierung abgeschlossen: {txt_file_path}")
                except Exception as e:
                    print(f"Fehler bei der Konvertierung von {pdf_file_path}: {str(e)}")

    print("Alle PDF-Dateien wurden verarbeitet.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Verwendung: python pdf2txt-recursiv.py <Quellverzeichnis> <Zielverzeichnis>")
        sys.exit(1)

    source_directory = sys.argv[1]
    target_directory = sys.argv[2]

    if not os.path.exists(source_directory):
        print(f"Fehler: Das Quellverzeichnis '{source_directory}' existiert nicht.")
        sys.exit(1)

    convert_pdf_to_txt(source_directory, target_directory)
