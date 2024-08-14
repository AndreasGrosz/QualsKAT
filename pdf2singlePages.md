# PDF Splitter und Konverter

Dieses Projekt enthält ein Skript, das PDF-Dateien in einzelne Seiten aufteilt und anschließend jede Seite in eine TXT-Datei konvertiert. Das Skript ist in Python geschrieben und nutzt die `PyPDF2`-Bibliothek.

## Anforderungen

Stellen Sie sicher, dass Python installiert ist. Zusätzlich benötigen Sie die `PyPDF2`-Bibliothek. Sie können sie über `pip` installieren:

```bash
pip install PyPDF2
Verwendung
PDF-Dateien aufteilen:

Das Skript durchsucht den aktuellen Ordner nach PDF-Dateien und erstellt für jede PDF-Datei einen neuen Ordner, in dem die einzelnen Seiten der PDF als separate Dateien gespeichert werden. Die Seiten werden im Format page_XXXX.pdf gespeichert, wobei XXXX die vierstellige Seitennummer ist (z.B. page_0001.pdf).

Konvertierung der gesplitteten PDFs zu TXT:

Nachdem die PDF-Dateien aufgeteilt wurden, konvertiert das Skript jede einzelne Seite in eine TXT-Datei. Der extrahierte Text jeder Seite wird in einer entsprechenden TXT-Datei gespeichert, die im selben Ordner wie die gesplitteten PDF-Dateien liegt.

## Ausführung
Führen Sie das Skript in dem Verzeichnis aus, das die zu verarbeitenden PDF-Dateien enthält:

bash

python script_name.py
Hinweis: Ersetzen Sie script_name.py durch den Namen des kombinierten Skripts.

## Dateistruktur
Nach der Ausführung des Skripts sieht die Verzeichnisstruktur wie folgt aus:

bash

/your-directory
    /example.pdf
    /example
        /example_page_0001.pdf
        /example_page_0002.pdf
        ...
        /example_page_0001.txt
        /example_page_0002.txt
        ...

## Fehlerbehandlung
Das Skript enthält einfache Fehlerbehandlungen. Sollte ein Fehler bei der Verarbeitung einer PDF-Datei auftreten, wird eine entsprechende Meldung im Terminal ausgegeben, und das Skript fährt mit der nächsten Datei fort.

## Lizenz
Dieses Projekt steht unter der MIT-Lizenz. Weitere Informationen finden Sie in der LICENSE-Datei.

## Urheber
Andreas Gross, Morgarten in Switzerland

Diese `README.md` bietet eine grundlegende Anleitung zur Installation, Verwendung und Dateistruktur des Projekts. Anpassungen können je nach spezifischen Anforderungen vorgenommen werden.


