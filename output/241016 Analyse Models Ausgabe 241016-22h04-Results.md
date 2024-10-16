# 241016 Analyse Models Ausgabe 241016-22h04-Results.md
## Deskriptive Statistik:
       albert-base-v2  distilbert-base-uncased  microsoft  roberta-base  roberta-large  Mittelwert
count         6808.00                  6808.00    6808.00       6808.00        6808.00     6808.00
mean             2.16                     2.48       1.21          4.46          23.46        6.51
std             12.56                     8.68       9.45         17.74          39.44       12.04
min              0.00                     0.00       0.00          0.00           0.00        0.00
25%              0.00                     0.00       0.00          0.00           0.00        0.00
50%              0.00                     1.00       0.00          0.00           0.00        0.00
75%              0.00                     1.00       0.00          0.00          30.00        9.00
max             99.00                    99.00      99.00         99.00         100.00       99.00


## Korrelationsmatrix:
                         albert-base-v2  distilbert-base-uncased  microsoft  roberta-base  roberta-large
albert-base-v2                     1.00                     0.14       0.20          0.27           0.23
distilbert-base-uncased            0.14                     1.00       0.15          0.39           0.26
microsoft                          0.20                     0.15       1.00          0.05           0.11
roberta-base                       0.27                     0.39       0.05          1.00           0.44
roberta-large                      0.23                     0.26       0.11          0.44           1.00


## Prozentsatz der konsistenten Vorhersagen: 69.33%


## Interessante Fälle (große Diskrepanz zwischen Modellen):
### Dokument: 59XXXX — HCO Bulletin — Page 2 Only  [B023-249].txt
  albert-base-v2            22
  distilbert-base-uncased            4
  microsoft         98
  roberta-base      66
  roberta-large     99

### Dokument: 5304XX — HCO Bulletin — Admiration Processing  [B017-015].txt
  albert-base-v2            99
  distilbert-base-uncased           12
  microsoft         93
  roberta-base       1
  roberta-large     99

### Dokument: 560803 — HCO Bulletin — Organizational Health Chart  [B033-012].txt
  albert-base-v2            90
  distilbert-base-uncased            1
  microsoft          1
  roberta-base      34
  roberta-large     99

### Dokument: 560912 — HCO Technical Bulletin — Summary of a Bulletin from the Academy in Washington, DC, Concerning Training, The  [B020-016].txt
  albert-base-v2             0
  distilbert-base-uncased            1
  microsoft          0
  roberta-base      92
  roberta-large     67

### Dokument: 560920 — HCO Processing Sheet — HCO Processing Sheet  [B020-017].txt
  albert-base-v2             0
  distilbert-base-uncased            1
  microsoft          0
  roberta-base      19
  roberta-large     99

## Modellperformanz-Übersicht:
        albert-base-v2  distilbert-base-uncased  microsoft  roberta-base  roberta-large  Mittelwert
mean              2.16                     2.48       1.21          4.46          23.46        6.51
median            0.00                     1.00       0.00          0.00           0.00        0.00
std              12.56                     8.68       9.45         17.74          39.44       12.04
min               0.00                     0.00       0.00          0.00           0.00        0.00
max              99.00                    99.00      99.00         99.00         100.00       99.00
## Prozentsatz der Dokumente über Schwellenwerten:
Threshold   albert-base-v2distilbert-base-uncased   microsoftroberta-baseroberta-large  Mittelwert
------------------------------------------------------------------------------------
>50%                1.81        1.03        1.07        4.19       23.16        1.22
>60%                1.69        0.87        1.00        3.69       22.31        0.25
>70%                1.62        0.60        0.87        3.26       21.43        0.09
>80%                1.45        0.43        0.81        2.82       20.48        0.04
>90%                1.25        0.24        0.68        2.13       18.79        0.04
