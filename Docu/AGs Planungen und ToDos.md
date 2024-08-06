# AGs Planungen und ToDos 

## AGs Workflow und GIThub

Ihre Überlegungen sind sehr gut. Lassen Sie uns eine optimale .gitignore-Datei und eine Strategie für die verschiedenen Ordner erstellen:

1. .gitignore Datei:  

Erstellen Sie eine .gitignore Datei im Root-Verzeichnis Ihres Projekts mit folgendem Inhalt:

```
Copy
# Ignore large data folders
/Documents/

# Ignore output and model folders
/output/
/models/

# Ignore Python cache files
__pycache__/
*.py[cod]

# Ignore Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Ignore environment-specific files
.env
venv/
*.log

# Ignore OS-specific files
.DS_Store
Thumbs.db
```

1. Ordner-Strategie:  

- Documents:  
  - Nicht in Git 	versionieren (zu groß)  	
  - Manuell zu Google 	Drive hochladen  	
  - In .gitignore 	aufnehmen  	
- CheckThis:  
  - In Git versionieren  	
  - Wird automatisch mit 	GitHub synchronisiert  	
  - Nicht in .gitignore 	aufnehmen  	
- output:  
  - Nicht in Git 	versionieren (ändert sich häufig und ist reproduzierbar)  	
  - In .gitignore 	aufnehmen  	
  - Lokal und in Google 	Drive beibehalten  	
- models:  
  - Nicht in Git 	versionieren (oft zu groß)  	
  - In .gitignore 	aufnehmen  	
  - Manuell zu Google Drive hochladen, wenn 	nötig  	

1. Projektstruktur:  

```
Copy
QualsKAT/
│
├── .gitignore
├── classifier.py
├── README.md
│
├── Documents/  (nicht in Git, manuell zu Drive)
│   └── ...
│
├── CheckThis/  (in Git)
│   └── ...
│
├── output/  (nicht in Git, lokal und in Drive)
│   └── ...
│
└── models/  (nicht in Git, manuell zu Drive wenn nötig)
    └── ...
```

1. Workflow:  

- Lokale Entwicklung:  
  - Arbeiten Sie an 	`classifier.py` und anderen Skripten  	
  - Testen Sie mit 	lokalen Daten  	
  - Committen und pushen 	Sie Änderungen zu GitHub  	
- GitHub:  
  - Enthält 	`classifier.py`, `README.md`, 	`.gitignore`, und den `CheckThis` 	Ordner  	
  - Dient als zentrale 	Versionskontrolle  	
- Google Drive:  
  - Enthält das gesamte 	Projekt, einschließlich großer Datensätze  	
  - `Documents`, 	`output`, und `models` 	Ordner werden manuell synchronisiert  	
  - Der Rest wird über 	Git synchronisiert  	
- Colab:  
  - Greift auf Dateien in 	Google Drive zu  	
  - Führt bei jedem Start ein `git 	pull` aus, um `classifier.py` und 	`CheckThis` zu aktualisieren  	

Mit dieser Struktur und Strategie:

- Halten Sie Ihr GitHub-Repository schlank und effizient  
- Behalten Sie die Kontrolle über große Dateien und Ausgaben  
- Ermöglichen Sie eine einfache Zusammenarbeit und Versionskontrolle für den Kerncode  
- Stellen Sie sicher, dass Colab immer mit den neuesten Versionen arbeitet

## Hilfreiche Kommandos

#### Alle *.doc eines Folders recursiv zu *.txt konvertieren:

find . -iname '*.doc' -exec sh -c 'echo "Konvertiere 
 $1"; libreoffice --headless --convert-to txt:Text --outdir "$(dirname "$1")" "$1" && echo "Fertig: $1"' sh {} \;
 
 Das geht auch mit anderen Formaten, die LO-Writer unterstützt.

#### files.py finden, die suchbegriff enthalten

grep -n -r "analyze_new_article" .

Oder für ein paar Zeilen:

grep -n -r "def analyze_new_article" . | xargs -I {} sh -c 'echo {}; sed -n "$(echo {} | cut -d: -f2),+5p" "$(echo {} | cut -d: -f1)"'

 ften copied after something he took from the entities. He found an entity role would

restimulate, he became the actor and performed the role. He left his own bank alone and neglected although there were aberrations to dramatize there too. (And by the way, you will find the thetan occasionally trying to stop the body from dramatizing out of entity banks).

The thetan bank, the one you want, will give

## ToDos

- ~~Alle Ausgaben von Wahrscheinlichkeiten in %, statt 0.nn~~
- verkürzen des param --checkthis auf --check
- checking der docs in alphab Reihenfolge
- wieso wird HoM ch 10 nicht als LRH-books erkannt, wo das HoM im Training ist. KDiff erkennt das chapter.
- checks zu kleiner Texte mit Warnung versehen: zu vage Ergebnisse f e Aussage
- die gpt-ausführungen zur textanalyse, linguistic speichern und recherchieren. 
- Serviceanbieter zur Textanalyse suchen und lernen und testen. z.B. den Anbieter für Unis via e befreundeten Professor.
- die einzelergebnisse gehören auch in die Logfile
- kann ich mit der logging lib in py z.B. alle debug ausgaben an zentraler stelle ein- und ausschalten?
- die schlussfolgerungen korrigieren. Ein Autor ist erst dann wahrscheinlich, wenn p>50% und unbekannt, wenn p<
- Vor python main.py --train –quick fragen, ob die Modelle überschrieben werdne dürfen.
- [~~CheckThisResults.csv~~](https://github.com/AndreasGrosz/QualsKAT/blob/main/CheckThisResults.csv)~~ ~~~~sollte nicht überschrieben werden, sondern angehängt, mit Datum~~
- [CheckThisResults.csv](https://github.com/AndreasGrosz/QualsKAT/blob/main/CheckThisResults.csv) als ods ausgeben m den gewünschten formatierungen.
- [CheckThisResults.csv](https://github.com/AndreasGrosz/QualsKAT/blob/main/CheckThisResults.csv) Datumsangaben ist nicht im Standard 01-08-24 16:56
- ~~python main.py –help – für hilfetext der parameter~~
- ​	kommentieren jeder Funktion
- Wie die OT-Materialien behandeln?
- FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 珞 Transformers. Use `eval_strategy` instead
- ​	Scriptausgabe des Trainingsergebnisses besser lesbar drucken: [file:///media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/Docu/240801%20Ergebnisbericht%20bis%2088%25%20Genauigkeit.odt#1.1.Traingsergebnis 	der Vollversion|outline]()
- ​	~~Trainingsdauer im Terminal ausgeben.~~
- ​	
   
   
   
   	
