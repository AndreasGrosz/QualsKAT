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

 

### 2. 8. 24 um 18h

Das script ist für lokale production schon geeignet. Es gibt Wahrscheinlichkeitswerte (prob) für LRH oder Ghosts, leider ist die Gesamtschlussfolgerung noch falsch: bei 8% LRH-prob und 6% ghosts-prob wird der Squirrel-Text LRH zugeordnet. Aber das lässt sich durch eine Schwellenwert noch beheben. Es gibt ja – wie dies Bsp zeigt - noch eine 3 Möglichkeit: der ghost ist hier nicht modelliert worden. Das wird bei den meisten Squirrel-Issues der fall sein.

Next step: Sammeln vieler Texte, die zu analysieren sind und Erfahrungen damit sammeln.

### 2. 8. 24 um 23h

Weiter verbessert. Die probab sind in %, wir haben max, mittel und sigma werte und so wird der beste guess geschätzt.

CBR ist sehr häufig als schätzung abgegeben. Das lag wohl daran, dass die meisten seiner files in unterfoldern waren, die nicht trainiert wurden. Das habe ich behoben. Vorher 14 files mit 1.2MB, jetzt für CBR 351 files mit 5.5MB

Nur wirkliche LRH-texte (incl. ME) werden mit über 

### 3. 8. 24 um 1h

--train mit diesen Autoren

1. Andreas Gross
2. Brian Livingston Cl. XII
3. CBR
4. Dan Koon
5. David Mayo  Cl. XII
6. Jeff Walker  Cl. XII
7. Jesse Prince
8. Jill Steinberg Cl. XII
9. Jim Dincalci Rons Medical Researcher
10. Joan Robertson, CS-4
11. John Eastment Cl. XII
12. Julie Gillespie
13. Julie Mayo
14. Ken Urquhart
15. Kima Jason, Snr MO Flag
16. LRH
17. Merril Mayo
18. Mike Goldstein
19. Mike Rinder
20. Molly Harlow
21. Otto J. Roos - OJP
22. Pat Brice LRH Comps Unit IC
23. Paulette Ausley, LRH Tech Expeditor
24. Paulette Cohen
25. Rick Sheehy
26. Ron Shaffran, CS-4 Class XII
27. RVY
28. Tony Dunleavy, CS 2
29. unknown

### Beobachtung zu d OT-Materialien

ich habe ein check auf OT I u III gemacht und verwirrende Bewertungen erhalten, das meiste sei von CBR. Das liegt wohl daran, dass die Nomenklatur f OT-Levels nicht im LRH-train-Daten enthalten sind, jedoch in denen von CBR. Dafür braucht es eine Lösung, weil gerade bei den OT-Materialien interessieren uns die Fälschungen.

### 4. 8. 24 um 1h

Neu trainiert, wobei ich LRH-Tech-Conf mit den **Class VIII lectures** zugefügt hatte.

Training mit 2 models: 

##### distilbert-base-uncased, Dauer 104 min

Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight'] 
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference. 
Map: 100%|████████████████████████████████████████| 1759/1759 [00:14<00:00, 122.52 examples/s] 
Map: 100%|███████████████████████████████████████████| 378/378 [00:04<00:00, 78.49 examples/s] 
Map: 100%|███████████████████████████████████████████| 377/377 [00:03<00:00, 94.30 examples/s] 
/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 珞 Transformers. Use `eval_strategy` instead   warnings.warn( 
{'eval_loss': 0.9570627212524414, 'eval_accuracy': 0.8015872836112976, 'eval_runtime': 124.6908, 'eval_samples_per_second': 3.031, 'eval_steps_per_second': 0.192, 'epoch': 1.0}       {'eval_loss': 0.6623812913894653, 'eval_accuracy': 0.8492063283920288, 'eval_runtime': 124.7018, 'eval_samples_per_second': 3.031, 'eval_steps_per_second': 0.192, 'epoch': 2.0}       {'eval_loss': 0.6067764163017273, 'eval_accuracy': 0.8544973731040955, 'eval_runtime': 97.2897, 'eval_samples_per_second': 3.885, 'eval_steps_per_second': 0.247, 'epoch': 3.0}        {'train_runtime': 6115.5447, 'train_samples_per_second': 0.863, 'train_steps_per_second': 0.054, 'train_loss': 0.9982582785866477, 'epoch': 3.0}                       
100%|█████████████████████████████████████████████████████| 330/330 [1:41:55<00:00, 18.53s/it] 
100%|█████████████████████████████████████████████████████████| 24/24 [01:34<00:00,  3.95s/it] 
(

###### albert-base-v2,  Dauer  182 min

Some weights of AlbertForSequenceClassification were not initialized from the model checkpoint at albert-base-v2 and are newly initialized: ['classifier.bias', 'classifier.weight'] 
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference. 
Map: 100%|█████████████████████████████████████████| 1759/1759 [00:19<00:00, 92.03 examples/s] 
Map: 100%|███████████████████████████████████████████| 378/378 [00:05<00:00, 66.67 examples/s] 
Map: 100%|███████████████████████████████████████████| 377/377 [00:04<00:00, 82.34 examples/s] 
/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 珞 Transformers. Use `eval_strategy` instead   warnings.warn( 
{'eval_loss': 0.6155022978782654, 'eval_accuracy': 0.8518518805503845, 'eval_runtime': 226.9118, 'eval_samples_per_second': 1.666, 'eval_steps_per_second': 0.106, 'epoch': 1.0}       {'eval_loss': 0.5441100001335144, 'eval_accuracy': 0.8650793433189392, 'eval_runtime': 226.4217, 'eval_samples_per_second': 1.669, 'eval_steps_per_second': 0.106, 'epoch': 2.0}       {'eval_loss': 0.5014544129371643, 'eval_accuracy': 0.8835979104042053, 'eval_runtime': 226.2636, 'eval_samples_per_second': 1.671, 'eval_steps_per_second': 0.106, 'epoch': 3.0}       {'train_runtime': 10664.5938, 'train_samples_per_second': 0.495, 'train_steps_per_second': 0.031, 'train_loss': 0.7111190333510891, 'epoch': 3.0}

Claude fasst die Ergebnisse zusammen:Die Zusammenfassung des Trainings mit den beiden Modellen "distilbert-base-uncased" und "albert-base-v2" enthält einige wichtige Informationen:

1. Trainingsdauer

   :

   - "distilbert-base-uncased" hatte eine Trainingsdauer von etwa 104 Minuten.
   - "albert-base-v2" hatte eine Trainingsdauer von etwa 182 Minuten.
   - Der deutliche Unterschied in der Trainingsdauer deutet darauf hin, dass das "albert-base-v2"-Modell deutlich komplexer und rechenintensiver ist als das "distilbert-base-uncased"-Modell.

2. Nicht initialisierte Gewichte

   :

   - Beide Modelle haben Gewichte, die nicht aus den Modell-Checkpoints geladen werden konnten und stattdessen neu initialisiert werden mussten.
   - Dies deutet darauf hin, dass die Modelle möglicherweise noch weiter auf einer spezifischen Aufgabe trainiert werden müssen, um für Vorhersagen und Inferenz verwendet werden zu können.

3. Trainings- und Evaluationsergebnisse

   :

   - Für beide Modelle werden die Verluste und Genauigkeiten für die einzelnen Trainingsepochen ausgegeben.
   - Diese Informationen geben einen Einblick in den Lernfortschritt und die Leistung der Modelle während des Trainings.

4. Fortschrittsbalken

   :

   - Die Fortschrittsbalken zeigen den Verlauf des Trainings und der Evaluation für beide Modelle.
   - Diese visuellen Indikatoren ermöglichen es, den Trainingsfortschritt auf einen Blick zu erfassen.

Insgesamt liefert diese Zusammenfassung wichtige Erkenntnisse über die Komplexität und Leistung der beiden Modelle. Die Unterschiede in Trainingsdauer und Leistung können dazu verwendet werden, die Modelle weiter zu optimieren und die am besten geeignete Konfiguration für deine Anwendung zu finden.

 CopyRetry



AG

welche genauen parameter ergeben sich durch diesen trainingsbericht?

 Edit

Aus den Trainings- und Evaluationsergebnissen, die in dem Bericht aufgeführt sind, lassen sich folgende Parameter für die beiden Modelle ableiten:

Für das **distilbert-base-uncased**-Modell:

- Trainingsdauer: 104 Minuten
- Trainings-Verlust (nach 3 Epochen): 0.9982582785866477
- Evaluations-Verlust (nach 3 Epochen): 0.6067764163017273
- Evaluations-Genauigkeit (nach 3 Epochen): 0.8544973731040955
- Evaluations-Samples pro Sekunde: 3.885
- Evaluations-Schritte pro Sekunde: 0.247

Für das **albert-base-v2**-Modell:

- Trainingsdauer: 182 Minuten
- Trainings-Verlust (nach 3 Epochen): 0.7111190333510891
- Evaluations-Verlust (nach 3 Epochen): 0.5014544129371643
- Evaluations-Genauigkeit (nach 3 Epochen): 0.8835979104042053
- Evaluations-Samples pro Sekunde: 1.671
- Evaluations-Schritte pro Sekunde: 0.106

Diese Kennzahlen geben dir einen guten Überblick über die Leistung und Effizienz der beiden Modelle. Anhand dieser Informationen kannst du zum Beispiel:

- Die Modelle miteinander vergleichen und die Vor- und Nachteile abwägen.
- Entscheiden, welches Modell für deine Anwendung am besten geeignet ist.
- Einschätzen, ob weitere Optimierungen oder Feinabstimmungen der Modelle sinnvoll sind.
- Abschätzen, wie viel Rechenleistung und Zeit für das Training der Modelle benötigt wird.

Mit diesen detaillierten Kennzahlen hast du eine solide Grundlage, um informierte Entscheidungen über den Einsatz der Modelle in deiner Anwendung zu treffen.

 
## 9.8.24 Vergleich der Models auf dem Power-PC

Basierend auf den vollständigen Trainingsergebnissen können wir nun eine detailliertere Analyse für jedes Modell durchführen. Hier ist eine Zusammenfassung der Ergebnisse für alle fünf Modelle:

DistilBERT (distilbert-base-uncased):

Eval Loss: 0.03939051181077957
Trainingszeit: 2.28 Minuten
Samples/Sekunde: 211.847


RoBERTa Large (roberta-large):

Eval Loss: 0.03179258108139038
Trainingszeit: 15.85 Minuten
Samples/Sekunde: 32.42


RoBERTa Base (roberta-base):

Eval Loss: 0.060943350195884705
Trainingszeit: 20.48 Minuten
Samples/Sekunde: 100.13


DeBERTa Base (microsoft/deberta-base):

Eval Loss: 0.03793901205062866
Trainingszeit: 28.59 Minuten
Samples/Sekunde: 58.281


ALBERT (albert-base-v2):

Eval Loss: 0.04202970862388611
Trainingszeit: 31.97 Minuten
Samples/Sekunde: 92.999



Auswertung:

Beste Genauigkeit (niedrigster Eval Loss):

RoBERTa Large (0.03179)
DeBERTa Base (0.03794)
DistilBERT (0.03939)
ALBERT (0.04203)
RoBERTa Base (0.06094)


Schnellstes Training:

DistilBERT (2.28 Minuten)
RoBERTa Large (15.85 Minuten)
RoBERTa Base (20.48 Minuten)
DeBERTa Base (28.59 Minuten)
ALBERT (31.97 Minuten)


Höchste Verarbeitungsgeschwindigkeit (Samples/Sekunde):

DistilBERT (211.847)
RoBERTa Base (100.13)
ALBERT (92.999)
DeBERTa Base (58.281)
RoBERTa Large (32.42)



Fazit:

RoBERTa Large zeigt die beste Genauigkeit, ist aber relativ langsam in der Verarbeitung.
DistilBERT bietet eine gute Balance zwischen Genauigkeit und Geschwindigkeit. Es ist das schnellste Modell mit der höchsten Verarbeitungsgeschwindigkeit und erreicht die drittbeste Genauigkeit.
DeBERTa Base zeigt die zweitbeste Genauigkeit, ist aber vergleichsweise langsam im Training.
ALBERT und RoBERTa Base zeigen etwas schlechtere Genauigkeit, wobei RoBERTa Base die schlechteste Leistung in Bezug auf den Eval Loss zeigt.

Empfehlung:
Für Ihr Ziel, zwischen LRH-Texten und Nicht-LRH-Texten zu unterscheiden, würde ich folgende Empfehlungen geben:

Wenn Genauigkeit absolute Priorität hat und Rechenzeit kein Problem ist, wählen Sie RoBERTa Large.
Für den besten Kompromiss zwischen Genauigkeit und Geschwindigkeit empfehle ich DistilBERT. Es bietet eine sehr gute Genauigkeit bei deutlich kürzerer Trainingszeit und höherer Verarbeitungsgeschwindigkeit.
Wenn Sie etwas mehr Genauigkeit als DistilBERT möchten und bereit sind, mehr Rechenzeit zu investieren, ist DeBERTa Base eine gute Wahl.

Letztendlich hängt die Wahl des besten Modells von Ihren spezifischen Anforderungen an Genauigkeit, Trainingszeit und Inferenzgeschwindigkeit ab. DistilBERT scheint jedoch in diesem Fall eine besonders attraktive Option zu sein, da es eine sehr gute Leistung bei deutlich geringerem Ressourcenaufwand bietet.


## 9.8.24-2 Vergleich der Models auf dem Power-PC
Ich habe die Trainingsdokumente bereinigt und nochmal einen Train-Lauf gemacht:

hier das Ergebnis: Kannst Du mir das tabellarisch darstellen:Certainly! I'll create a table summarizing the fine-tuning results for the five models based on the information provided. Here's a tabular representation of the results:

| Model            | Train Loss | Train Runtime | Train Samples/s | Eval Loss | Eval Runtime |
|------------------|------------|---------------|-----------------|-----------|--------------|
| DistilBERT       |     6.6%   |     2.1 min   |      41.3       |    1.0%   |   0.0 min    |
| RoBERTa (large)  |     5.4%   |    13.1 min   |       6.7       |    0.1%   |   0.2 min    |
| RoBERTa (base)   |     7.1%   |     4.4 min   |      20.1       |    0.8%   |   0.1 min    |
| DeBERTa          |     5.6%   |     7.8 min   |      11.3       |    2.4%   |   0.1 min    |
| ALBERT           |     5.4%   |     3.1 min   |      27.9       |    0.1%   |   0.1 min    |

Notes:

The evaluation loss reported is from the last evaluation epoch for each model.
Train and eval runtimes are in seconds.
The "Train Samples/s" column represents the number of training samples processed per second.

This table allows for easy comparison of the models' performance and efficiency. RoBERTa (large) had the lowest eval loss but took the longest to train, while ALBERT achieved a good balance between low eval loss and fast training time. CopyRetryAGbitte weiter auf Deutsch. Was bedeutet eval loss für meine zwecke EditEntschuldigung für den Wechsel ins Englische. Ich fahre auf Deutsch fort.
Der "eval loss" (Evaluierungsverlust) ist für Ihre Zwecke sehr wichtig. Er gibt an, wie gut das Modell auf Daten abschneidet, die es während des Trainings nicht gesehen hat. Ein niedrigerer eval loss deutet auf eine bessere Generalisierungsfähigkeit des Modells hin. Das bedeutet:

Bessere Klassifizierung: Das Modell wird wahrscheinlich neue, ungesehene Dokumente genauer als "Autor 0" oder "Anderes" einordnen können.
Weniger Overfitting: Ein niedriger eval loss im Vergleich zum Trainingsverlust zeigt, dass das Modell nicht nur die Trainingsdaten auswendig gelernt hat, sondern tatsächlich relevante Merkmale erfasst hat.
Robustheit: Modelle mit niedrigerem eval loss sind oft robuster gegenüber leichten Variationen in den Eingabedaten.
Vertrauenswürdigkeit: Sie können mehr Vertrauen in die Vorhersagen des Modells haben, wenn es einen niedrigen eval loss aufweist.

Basierend auf den eval loss Werten scheinen RoBERTa (large) und ALBERT am besten zu generalisieren, gefolgt von DistilBERT. Diese Modelle könnten für Ihre Aufgabe der Autorenerkennung am geeignetsten sein.
