# Entwicklungslog

Start: 31.7.24 um 20h44

\>python main.py --train --quick
 ist erfolgreich. Git add etc machen

 

Folgende Varianten noch austesten:

\>python main.py 	# f help

\>python main.py --checkthis --quick

\>python main.py --train --quick

\>python main.py --train 

\>python main.py --checkthis 




### 1.8.24 12h

python main.py --train 
 funkte, siehe [/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/Docu/240801 Ergebnisbericht bis 88% Genauigkeit.odt]()

Unbekannte Labels: Der Hauptfehler, der für alle Dateien auftritt, ist: "y contains previously unseen labels". Dies bedeutet, dass die Modelle auf Labels (Kategorien) stoßen, die sie während des Trainings nicht gesehen haben.

##### Lösung 12h22

Mit diesen Änderungen:

1. Sie können `update_categories.py` ausführen, um die Kategorien basierend auf der Verzeichnisstruktur zu extrahieren und in einer CSV-Datei zu speichern.  
2. Das Hauptskript verwendet dann diese vordefinierten Kategorien für das Training und die Vorhersagen.  
3. Die Logik für die Kategorisierung bleibt konsistent zwischen der Kategorie-Extraktion und dem Training/Vorhersage-Prozess.  

Um das System zu aktualisieren:

1. Führen Sie `python update_categories.py` aus, wenn Sie neue Daten hinzufügen oder die Verzeichnisstruktur ändern.  
2. Führen Sie dann `python main.py` mit den gewünschten Optionen aus.  

Diese Methode stellt sicher, dass Sie die gleiche Kategorisierungslogik für das Erstellen der CSV und das Training verwenden, während Sie auch die Flexibilität haben, die Kategorien bei Bedarf zu aktualisieren.

### 15:17 Kategorien säubern

Noch nicht gemacht, erst vor dem nächsten Volltraining

Bezüglich der Kategorien in `categories.csv`: Es sieht so aus, als hätten wir einige Probleme mit der Formatierung und unerwünschten Zeichen in den Kategorienamen. Wir sollten dies bereinigen. 

1. Der Zeilenumbruch im Ordnernamen von "Ghostwriter-Jim Dincalci": Sie haben Recht, dass wir dies vor dem nächsten Training korrigieren sollten. Hier sind die Schritte: a. Korrigieren Sie den Ordnernamen manuell. b. Führen Sie `update_categories.py` aus, um die `categories.csv` zu aktualisieren. c. Dann führen Sie `python main.py --train --quick` aus.  

Um die Kategorien zu bereinigen, können wir eine Funktion in `update_categories.py` hinzufügen:

```
python
Copy
import re

def clean_category_name(category):
    # Entfernt Anführungszeichen am Anfang und Ende
    category = category.strip('"')
    # Ersetzt mehrere Leerzeichen durch ein einzelnes
    category = re.sub(r'\s+', ' ', category)
    # Entfernt Zeilenumbrüche
    category = category.replace('\n', '')
    return category.strip()

def update_categories(config_path):
    # ... (vorheriger Code bleibt unverändert)
    
    categories = extract_categories(config)
    categories = [clean_category_name(cat) for cat in categories]
    save_categories_to_csv(categories, config)
```

Nach diesen Änderungen und der manuellen Korrektur des Ordnernamens sollten Sie folgende Schritte ausführen:

1. Korrigieren Sie den Ordnernamen für "Ghostwriter-Jim Dincalci".  
2. Führen Sie `python update_categories.py` aus.  
3. Überprüfen Sie die aktualisierte `categories.csv`.  
4. Führen Sie `python main.py --train --quick` aus.  

Diese Schritte sollten die Kategorien bereinigen und das Problem mit dem Zeilenumbruch beheben. Lassen Sie mich wissen, wenn Sie die Änderungen vorgenommen haben und wie das Ergebnis aussieht.

CopyRetry

Claude does not have the ability to run the code it generates yet.

AG

würde python main.py --train --quick nicht das grosse model überschreiben? das will ich möglichst eine weile bewahren.

Edit

Sie haben absolut Recht, und ich danke Ihnen für diese wichtige Beobachtung. Es stimmt, dass `python main.py --train --quick` das bestehende Modell überschreiben würde, was in Ihrem Fall unerwünscht ist, da Sie das vollständig trainierte Modell bewahren möchten.

Hier sind einige Optionen, wie wir vorgehen können, ohne das bestehende Modell zu überschreiben:

1. Temporäres Modell für Quick-Training: Wir können die `main.py` so anpassen, dass sie bei Verwendung der `--quick` Option ein separates, temporäres Modell erstellt und trainiert.  
2. Nur Vorhersagen ohne Training: Wenn Sie nur die Kategorien aktualisieren und dann Vorhersagen machen möchten, ohne neu zu trainieren, können wir einen neuen Modus einführen.  
3. Backup des bestehenden Modells: Vor dem Training könnte automatisch ein Backup des bestehenden Modells erstellt werden.  

Ich schlage vor, Option 2 zu implementieren, da sie am besten zu Ihrem aktuellen Bedarf passt. Hier sind die Änderungen, die wir vornehmen könnten:

1. In `main.py`, fügen Sie eine neue Option hinzu:  

```
python
Copy
parser.add_argument('--predict_only', action='store_true', 
                    help='Lädt das bestehende Modell und macht nur Vorhersagen, ohne zu trainieren.')
```

1. Ändern Sie den Hauptteil der `main()` Funktion:  

```
python
Copy
if args.train:
    # Bestehender Trainings-Code
    ...
elif args.predict_only:
    # Laden des bestehenden Modells
    model_save_path = os.path.join(config['Paths']['models'], model_name.replace('/', '_'))
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
    trainer = Trainer(model=model)

if args.checkthis or args.predict:
    # Bestehender Code für Vorhersagen
    ...
```

Mit diesen Änderungen können Sie folgende Schritte ausführen:

1. Korrigieren Sie den Ordnernamen für "Ghostwriter-Jim Dincalci".  
2. Führen Sie `python update_categories.py` aus, um die `categories.csv` zu aktualisieren.  
3. Führen Sie `python main.py --predict_only --checkthis` aus, um Vorhersagen mit dem bestehenden Modell und den aktualisierten Kategorien zu machen.  

Diese Vorgehensweise erlaubt es Ihnen, die Kategorien zu aktualisieren und neue Vorhersagen zu machen, ohne das bestehende, vollständig trainierte Modell zu verändern.

Wenn Sie in Zukunft ein neues Modell mit den aktualisierten Kategorien trainieren möchten, können Sie dann `python main.py --train` ausführen. Es wäre ratsam, vorher ein Backup des aktuellen Modells zu erstellen.

Lassen Sie mich wissen, ob dieser Ansatz für Sie funktioniert oder ob Sie weitere Anpassungen wünschen.

### 1.8.24 um 17h Beobachtungen von python main.py --checkthis 

#### LRH wurde immer auf 0 geschätzt, möglicherweise realistische Werte für die Ghostwriter. Wurde er überhaupt geprüft? In früheren Versionen waren seine Ergebnisse über 80% bei seinen Texten.

#### Alle Modelle in einer Schleife: es fehlt die Überschrift in der print-ausgabe, welches Modell gerade bearbeitet wird. Bitte in GRÜN hervorheben.     

#### Ebenso in der csv-Ausgabe, eine weitere Spalte D model wurde eingefügt, wo das model eingetragen wird.




### Das folgende in der Ausgabe sind wohl debug-zeilen, die wir nicht mehr brauchen, also löschen:

Spalten nach der Tokenisierung: 
 ['labels', 'input_ids', 'attention_mask'] 
 ['labels', 'input_ids', 'attention_mask'] 
 Struktur des tokenisierten Trainingsdatensatzes: 
 {'labels': Value(dtype='int64', id=None), 'input_ids': Sequence(feature=Value(dtype='int32', id=None
 ), length=-1, id=None), 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, 
 id=None)} 
 Struktur des tokenisierten Testdatensatzes: 
 {'labels': Value(dtype='int64', id=None), 'input_ids': Sequence(feature=Value(dtype='int32', id=None
 ), length=-1, id=None), 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, 
 id=None)}

\--------

In der Ausgabe gibt es diese HE: Debug - Vorhersagen: 

dahinter fehlt die Datei, die untersucht wird, sonst machen die Infos keinen Sinn.

#### Was bedeutet diese Meldung

Some weights of AlbertForSequenceClassification were not initialized from the model checkpoint at albert-base-v2 and are newly initialized: ['classifier.bias', 'classifier.weight'] 
 You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference. 

 

Weiter bei 4. von Claude Debug-Zeilen entfernen: 