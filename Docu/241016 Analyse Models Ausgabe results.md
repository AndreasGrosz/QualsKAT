# 241016 Analyse Models Ausgabe results.md



## Deskriptive Statistik:
         r-base  ms-deberta   distilb   r-large    albert
count  26088.00    26088.00  26088.00  26088.00  26088.00
mean      16.99       46.18     26.87     34.71     32.98
std       35.26       48.78     39.14     46.07     45.06
min        0.00        0.20      0.90      0.00      0.30
25%        0.10        0.20      1.20      0.00      0.50
50%        0.10        2.10      2.20      0.30      1.20
75%        1.70       99.80     43.40    100.00     99.90
max      100.00      100.00     99.90    100.00    100.00


## Korrelationsmatrix:
            r-base  ms-deberta  distilb  r-large  albert
r-base        1.00        0.30     0.60     0.61    0.52
ms-deberta    0.30        1.00     0.51     0.45    0.64
distilb       0.60        0.51     1.00     0.58    0.72
r-large       0.61        0.45     0.58     1.00    0.63
albert        0.52        0.64     0.72     0.63    1.00


## Prozentsatz der konsistenten Vorhersagen: 44.85%


## Interessante Fälle (große Diskrepanz zwischen Modellen):
### Dokument: 1950_53_Seite_537.txt
  r-base          1.20
  ms-deberta      0.20
  distilb         2.30
  r-large        92.80
  albert          0.60

### Dokument: 1950_53_Seite_26.txt
  r-base          2.00
  ms-deberta      2.20
  distilb         1.50
  r-large        92.10
  albert          1.20

### Dokument: 1950_53_Seite_526.txt
  r-base          0.40
  ms-deberta      0.20
  distilb         2.50
  r-large        97.70
  albert          0.60

### Dokument: 1950_53_Seite_542.txt
  r-base          0.40
  ms-deberta      0.20
  distilb         2.20
  r-large        97.90
  albert          1.40

### Dokument: 1950_53_Seite_523.txt
  r-base          0.80
  ms-deberta     94.90
  distilb         1.80
  r-large         4.30
  albert          0.80

## Prozentsatz der Dokumente über Schwellenwerten:
Threshold         r-base  ms-deberta     distilb     r-large      albert
------------------------------------------------------------------------
>50%               16.66       46.14       24.08       34.31       31.98
>60%               16.11       45.89       23.28       33.78       31.58
>70%               15.54       45.60       22.27       33.27       31.17
>80%               14.71       45.12       21.05       32.74       30.73
>90%               13.58       44.48       19.33       31.85       30.04

## Modellperformanz-Übersicht:
        r-base  ms-deberta  distilb  r-large  albert
mean     16.99       46.18    26.87    34.71   32.98
median    0.10        2.10     2.20     0.30    1.20
std      35.26       48.78    39.14    46.07   45.06
min       0.00        0.20     0.90     0.00    0.30
max     100.00      100.00    99.90   100.00  100.00
