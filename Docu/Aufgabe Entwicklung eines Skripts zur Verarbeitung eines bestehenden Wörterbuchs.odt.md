# Aufgabe Entwicklung eines Skripts zur Verarbeitung eines bestehenden Wörterbuchs

Eingabe:

-   Eine Textdatei im .txt-Format
-   Wörter sind durch Leerzeichen getrennt
-   Zeilenumbrüche werden ignoriert
-   Doppelwörter müssen nicht zusammenbleiben

Verarbeitung:

1.  Lesen der Eingabedatei

2.  Trennen der Wörter anhand von Leerzeichen

3.  Filtern der Wörter:

    -   Entfernen von Standard-Englischen Wörtern
    -   Beibehalten von Abkürzungen und kurzen Termen (≤ 3 Zeichen)
    -   Beibehalten von Wörtern mit Sonderzeichen (z.B. \'/\')

4.  Erstellen einer Liste spezialisierter Begriffe

Ausgabe:

-   Eine neue Textdatei mit den extrahierten spezialisierten Begriffen
-   Ein Begriff pro Zeile
-   Alphabetisch sortiert (unabhängig von Groß-/Kleinschreibung)

Zusätzliche Anforderungen:

-   Beibehaltung der Original-Groß-/Kleinschreibung
-   Effiziente Verarbeitung großer Wörterbücher

Diese Zusammenfassung kann als Grundlage für einen neuen Chat dienen, in
dem wir das Skript im Detail entwickeln und implementieren können.
