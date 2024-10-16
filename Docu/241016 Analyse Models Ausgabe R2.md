# 241016 Analyse Models Ausgabe HCOBs-analysis_results.md

## Deskriptive Statistik:
        r-base  ms-deberta  distilb  r-large   albert
count  6808.00     6808.00  6808.00  6808.00  6808.00
mean      4.63        1.47     2.95    23.75     2.62
std      17.83        9.49     8.68    39.68    12.60
min       0.00        0.20     0.80     0.00     0.30
25%       0.10        0.20     1.00     0.00     0.40
50%       0.10        0.20     1.10     0.10     0.40
75%       0.20        0.30     1.70    30.82     0.60
max     100.00      100.00    99.80   100.00   100.00

### Modellperformanz:

* **r-large** zeigt die höchste durchschnittliche Konfidenz (23.75%), gefolgt von r-base (4.63%).
* ms-deberta hat die niedrigste durchschnittliche Konfidenz (1.47%).
* Alle Modelle zeigen eine große Spannweite von 0 bis nahezu 100%, was auf sehr unterschiedliche Vorhersagen je nach Dokument hindeutet.

## Korrelationsmatrix:
            r-base  ms-deberta  distilb  r-large  albert
r-base        1.00        0.06     0.39     0.44    0.27
ms-deberta    0.06        1.00     0.15     0.12    0.21
distilb       0.39        0.15     1.00     0.26    0.14
r-large       0.44        0.12     0.26     1.00    0.23
albert        0.27        0.21     0.14     0.23    1.00
Die höchste Korrelation besteht zwischen **r-base und r-large (0.44)**, was darauf hindeutet, dass diese beiden Modelle ähnliche Muster in ihren Vorhersagen zeigen.
**ms-deberta** zeigt die geringsten Korrelationen mit den anderen Modellen, was auf ein möglicherweise **unabhängigeres Verhalten** hindeutet.

## Prozentsatz der konsistenten Vorhersagen: 68.91%
**68.91%** der Vorhersagen sind konsistent (Unterschied zwischen höchstem und niedrigstem Wert ≤ 10 Prozentpunkte).
Dies deutet darauf hin, dass die Modelle in etwa 2/3 der Fälle relativ ähnliche Einschätzungen liefern.

## Interessante Fälle (große Diskrepanz zwischen Modellen):
Es gibt mehrere Fälle, bei denen die Modelle stark voneinander abweichen (>90 Prozentpunkte Unterschied).
**r-large** neigt in diesen Fällen zu sehr hohen Werten (nahe 100%), während andere Modelle oft niedrigere Werte liefern.

### Dokument: 600129 — HCO Bulletin — Congresses  [B036-023].txt
  r-base: 32.00
  ms-deberta: 0.20
  distilb: 5.10
  r-large: 99.90
  albert: 1.00

### Dokument: 780711 — HCO Bulletin — Preassessment List, The  [B074-013].txt
  r-base: 0.90
  ms-deberta: 0.20
  distilb: 5.70
  r-large: 96.00
  albert: 0.50

### Dokument: 630901 — HCO Bulletin — Routine 3SC  [B137-004].txt
  r-base: 0.30
  ms-deberta: 3.10
  distilb: 2.80
  r-large: 99.90
  albert: 0.60

### Dokument: 611102 — HCO Bulletin — Prior Confusion, The  [B001-094].txt
  r-base: 0.60
  ms-deberta: 0.20
  distilb: 2.00
  r-large: 100.00
  albert: 0.60

### Dokument: 590718 — HCO Bulletin — Technically Speaking  [B113-009].txt
  r-base: 91.30
  ms-deberta: 0.20
  distilb: 82.30
  r-large: 100.00
  albert: 9.30

## Prozentsatz der Dokumente über Schwellenwerten:
Threshold         r-base  ms-deberta     distilb     r-large      albert
------------------------------------------------------------------------
>50%                4.21        1.09        1.06       23.23        1.83
>60%                3.76        1.01        0.88       22.41        1.72
>70%                3.32        0.87        0.62       21.48        1.61
>80%                2.89        0.81        0.47       20.56        1.48
>90%                2.17        0.69        0.25       19.03        1.31

Diese Tabelle bietet einen detaillierten Einblick in die Leistung der verschiedenen Modelle bei unterschiedlichen Konfidenz-Schwellenwerten. Hier sind einige wichtige Beobachtungen und Schlussfolgerungen:

### Konsistenz der Modelle:
r-large ist durchweg das "zuversichtlichste" Modell, mit dem höchsten Prozentsatz an Dokumenten über allen Schwellenwerten.
Die Reihenfolge der Modelle bleibt über alle Schwellenwerte konstant: r-large > r-base > albert > ms-deberta > distilb.

### Abnahme der Konfidenz:
Bei allen Modellen sinkt der Prozentsatz der Dokumente mit steigendem Schwellenwert, was zu erwarten ist.
Die Abnahme ist bei r-large am geringsten (von 23.23% auf 19.03%), was auf eine stabilere Konfidenz hindeutet.

### Unterschiede zwischen den Modellen:
Es gibt einen erheblichen Unterschied zwischen r-large und den anderen Modellen. Bei einem Schwellenwert von 50% klassifiziert r-large etwa 5-mal so viele Dokumente als "wahrscheinlich LRH" im Vergleich zum nächstbesten Modell (r-base).
Die Unterschiede zwischen ms-deberta, distilb und albert sind relativ gering, besonders bei höheren Schwellenwerten.

### Konservative Einschätzungen:
Mit Ausnahme von r-large klassifizieren alle Modelle weniger als 5% der Dokumente mit hoher Konfidenz (>50%) als von LRH stammend.
Selbst bei einem niedrigen Schwellenwert von 50% identifiziert das zweitbeste Modell (r-base) nur 4.21% der Dokumente als wahrscheinlich von LRH.

### Robustheit von r-large:
r-large behält einen relativ hohen Prozentsatz (19.03%) selbst bei einem sehr hohen Schwellenwert von 90%, was auf eine hohe Konfidenz in seine Klassifizierungen hindeutet.

### Sensitivität der Modelle:
distilb zeigt die größte relative Abnahme zwischen 50% und 90% (von 1.06% auf 0.25%), was auf eine höhere Sensitivität gegenüber dem Schwellenwert hindeutet.

### Schlussfolgerungen und nächste Schritte:

r-large scheint das vielversprechendste Modell zu sein, aber seine deutlich höheren Werte könnten auch auf Überconfidence hindeuten. Eine Überprüfung der von r-large hoch bewerteten Dokumente wäre sinnvoll.
Die Kombination von r-large mit einem oder mehreren der konservativeren Modelle könnte zu ausgewogeneren Ergebnissen führen.
Es wäre interessant zu untersuchen, warum r-large so viel "zuversichtlicher" ist als die anderen Modelle. Gibt es strukturelle Unterschiede im Modell oder in der Art, wie es trainiert wurde?
Angesichts der niedrigen Prozentwerte bei den meisten Modellen sollte überprüft werden, ob die Schwellenwerte angemessen sind oder ob eine Kalibrierung der Modelle erforderlich ist.
Eine Analyse der Dokumente, die von allen Modellen mit hoher Konfidenz klassifiziert wurden, könnte wertvolle Einblicke in die charakteristischen Merkmale von LRH's Schreibstil liefern.

Diese Ergebnisse unterstreichen die Herausforderungen bei der Autorschaftszuordnung und zeigen, wie wichtig es ist, multiple Modelle und Schwellenwerte zu betrachten, um ein umfassendes Bild zu erhalten.

## Modellperformanz-Übersicht:
        r-base  ms-deberta  distilb  r-large  albert
mean      4.63        1.47     2.95    23.75    2.62
median    0.10        0.20     1.10     0.10    0.40
std      17.83        9.49     8.68    39.68   12.60
min       0.00        0.20     0.80     0.00    0.30
max     100.00      100.00    99.80   100.00  100.00

Die **Medianwerte aller Modelle sind sehr niedrig (0.1-1.1%)**, was darauf hindeutet, dass die meisten Vorhersagen eher niedrige Konfidenzwerte haben.
Die hohen Durchschnittswerte im Vergleich zu den niedrigen Medianwerten deuten auf eine schiefe Verteilung hin, mit einigen sehr hohen Werten, die den Durchschnitt nach oben ziehen.
