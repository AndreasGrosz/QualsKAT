# 241016 Analyse Models Ausgabe.md

## Deskriptive Statistik:
            r-base   ms-deberta      distilb      r-large       albert
count  6808.000000  6808.000000  6808.000000  6808.000000  6808.000000
mean      4.626733     1.469477     2.954568    23.747518     2.624941
std      17.828034     9.488992     8.679646    39.681986    12.597483
min       0.000000     0.200000     0.800000     0.000000     0.300000
25%       0.100000     0.200000     1.000000     0.000000     0.400000
50%       0.100000     0.200000     1.100000     0.100000     0.400000
75%       0.200000     0.300000     1.700000    30.825000     0.600000
max     100.000000   100.000000    99.800000   100.000000   100.000000


## Korrelationsmatrix:
              r-base  ms-deberta   distilb   r-large    albert
r-base      1.000000    0.055156  0.393728  0.444574  0.274113
ms-deberta  0.055156    1.000000  0.149003  0.117176  0.206482
distilb     0.393728    0.149003  1.000000  0.261755  0.142579
r-large     0.444574    0.117176  0.261755  1.000000  0.232198
albert      0.274113    0.206482  0.142579  0.232198  1.000000


## Prozentsatz der konsistenten Vorhersagen: 68.91%


## Interessante Fälle (große Diskrepanz zwischen Modellen):
Dokument: 600129 — HCO Bulletin — Congresses  [B036-023].txt
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

## Modellperformanz-Übersicht:
                 Mean  Median        Std  Min    Max
r-base       4.626733     0.1  17.828034  0.0  100.0
ms-deberta   1.469477     0.2   9.488992  0.2  100.0
distilb      2.954568     1.1   8.679646  0.8   99.8
r-large     23.747518     0.1  39.681986  0.0  100.0
albert       2.624941     0.4  12.597483  0.3  100.0
