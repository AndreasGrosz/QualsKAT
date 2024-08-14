# CSV to ODS Converter mit funktionierenden Hyperlinks

## Übersicht
Dieses Python-Skript konvertiert eine CSV-Datei in ein ODS (OpenDocument Spreadsheet) Format mit zusätzlichen Funktionen wie Sortierung und funktionierenden Hyperlinks zu lokalen Dateien.

## Funktionen
- Liest eine 'ergebnisse.csv' Datei aus dem lokalen Verzeichnis
- Erstellt eine neue Spalte 'Sorted' mit umformatiertem Dateinamen
- Fügt eine Spalte 'Hyperlink' mit klickbaren Links zu den lokalen Dateien hinzu
- Sortiert die Daten basierend auf der 'Sorted'-Spalte
- Speichert das Ergebnis als 'ergebnisse.ods' im ODS-Format

## Anforderungen
- Python 3.x
- pandas
- odfpy

## Installation
1. Stellen Sie sicher, dass Python 3.x auf Ihrem System installiert ist.
2. Installieren Sie die erforderlichen Bibliotheken mit pip:
   ```
   pip install pandas odfpy
   ```

## Verwendung
1. Platzieren Sie die 'ergebnisse.csv' Datei im gleichen Verzeichnis wie das Skript.
2. Stellen Sie sicher, dass alle Textdateien, auf die in der CSV verwiesen wird, sich ebenfalls in diesem Verzeichnis befinden.
3. Führen Sie das Skript aus:
   ```
   python csv_to_ods_converter.py
   ```
4. Nach der Ausführung finden Sie die 'ergebnisse.ods' Datei im gleichen Verzeichnis.

## Wichtige Hinweise
- Das Skript verwendet das aktuelle Arbeitsverzeichnis, um die vollständigen Pfade für die Hyperlinks zu erstellen.
- Stellen Sie sicher, dass Sie das Skript aus dem Verzeichnis ausführen, in dem sich die Textdateien befinden.
- Die erzeugten Hyperlinks verwenden absolute Pfade. Wenn Sie die ODS-Datei auf einen anderen Computer übertragen, müssen die Hyperlinks möglicherweise angepasst werden.

## Dateiformat
Die Eingabe-CSV-Datei sollte folgendes Format haben:
```
"Filename","r-base","ms-deberta","distilb","r-large","albert","Mittelwert"
"V13 1985-1991_Seite_245.txt",0,0.3,1.2,0.1,0.6,0.4
"V13 1985-1991_Seite_260.txt",0.1,0.2,1.1,0,0.6,0.4
...
```

## Funktionsweise
- Das Skript geht davon aus, dass die Dateinamen im Format 'YYYY_YY_Seite_XXX.txt' vorliegen.
- Es konvertiert 'Seite_XXX' zu 'page_0XXX' für die Sortierung.
- Hyperlinks werden mit dem vollständigen Pfad erstellt: `file:///[vollständiger_pfad]/[dateiname]`

## Fehlerbehebung
Wenn Sie auf Probleme stoßen, stellen Sie sicher, dass:
1. Die 'ergebnisse.csv' Datei im korrekten Format vorliegt und lesbar ist.
2. Alle erforderlichen Bibliotheken korrekt installiert sind.
3. Sie über ausreichende Berechtigungen verfügen, um Dateien im Arbeitsverzeichnis zu lesen und zu schreiben.
4. Die Textdateien, auf die verwiesen wird, tatsächlich im angegebenen Verzeichnis existieren.
5. Ihr System korrekt konfiguriert ist, um .txt-Dateien mit dem gewünschten Texteditor zu öffnen.

Bei weiteren Problemen überprüfen Sie bitte die Fehlermeldungen und konsultieren Sie die Dokumentation der verwendeten Bibliotheken.

## Anpassungen
Wenn Sie den Speicherort der Textdateien ändern möchten, passen Sie die `create_hyperlink` Funktion im Skript an, um den korrekten Pfad zu verwenden.

## Wartung
Bei Änderungen an den Dateipfaden oder der Struktur Ihrer Daten, denken Sie daran, das Skript entsprechend anzupassen.
