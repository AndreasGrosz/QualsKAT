

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


## Korrelationsmatrix:
            r-base  ms-deberta  distilb  r-large  albert
r-base        1.00        0.06     0.39     0.44    0.27
ms-deberta    0.06        1.00     0.15     0.12    0.21
distilb       0.39        0.15     1.00     0.26    0.14
r-large       0.44        0.12     0.26     1.00    0.23
albert        0.27        0.21     0.14     0.23    1.00


## Prozentsatz der konsistenten Vorhersagen: 68.91%


## Interessante Fälle (große Diskrepanz zwischen Modellen):
Dokument: 600129 — HCO Bulletin — Congresses  [B036-023].txt
  r-base: 32.00
  ms-deberta: 0.20
  distilb: 5.10
  r-large: 99.90
  albert: 1.00

Dokument: 780711 — HCO Bulletin — Preassessment List, The  [B074-013].txt
  r-base: 0.90
  ms-deberta: 0.20
  distilb: 5.70
  r-large: 96.00
  albert: 0.50

Dokument: 630901 — HCO Bulletin — Routine 3SC  [B137-004].txt
  r-base: 0.30
  ms-deberta: 3.10
  distilb: 2.80
  r-large: 99.90
  albert: 0.60

Dokument: 611102 — HCO Bulletin — Prior Confusion, The  [B001-094].txt
  r-base: 0.60
  ms-deberta: 0.20
  distilb: 2.00
  r-large: 100.00
  albert: 0.60

Dokument: 590718 — HCO Bulletin — Technically Speaking  [B113-009].txt
  r-base: 91.30
  ms-deberta: 0.20
  distilb: 82.30
  r-large: 100.00
  albert: 9.30

## Modellperformanz-Übersicht:
        r-base  ms-deberta  distilb  r-large  albert
mean      4.63        1.47     2.95    23.75    2.62
median    0.10        0.20     1.10     0.10    0.40
std      17.83        9.49     8.68    39.68   12.60
min       0.00        0.20     0.80     0.00    0.30
max     100.00      100.00    99.80   100.00  100.00
