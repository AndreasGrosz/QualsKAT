"""
Dieses Script führt folgende Analysen durch:

Berechnet deskriptive Statistiken für jedes Modell.
Erstellt eine Korrelationsmatrix und visualisiert sie als Heatmap.
Analysiert die Konsistenz der Vorhersagen zwischen den Modellen.
Visualisiert die Verteilung der Vorhersagen für jedes Modell.
Identifiziert interessante Fälle, bei denen die Modelle stark voneinander abweichen.
Gibt eine Übersicht über die Performanz jedes Modells.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# Daten laden
# df = pd.read_csv('output/HCOBs-analysis_results.csv')

input_file = 'output/results-red-vols-1970s.csv'
df = pd.read_csv(input_file)

# Modellnamen
models = ['r-base', 'ms-deberta', 'distilb', 'r-large', 'albert']


# Überprüfen, ob alle Modellspalten numerisch sind
for model in models:
    if df[model].dtype == 'object':
        df[model] = pd.to_numeric(df[model], errors='coerce')


# Überschrift
# Aktuelles Datum
current_date = date.today().strftime("%y%m%d")
# Name der Eingabedatei ohne Pfad und Erweiterung
input_file_name = input_file.split('/')[-1].split('.')[0]
print(f"# {current_date} Analyse Models Ausgabe {input_file_name}.md")

# 1. Deskriptive Statistik
print("## Deskriptive Statistik:")
print(df[models].describe().round(2))
print("\n")

# 2. Korrelationsanalyse
corr = df[models].corr()
print("## Korrelationsmatrix:")
print(corr.round(2))
print("\n")

# Heatmap der Korrelationen
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Korrelation zwischen Modellen')
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Konsistenzanalyse
threshold = 10  # Schwellenwert für Übereinstimmung in Prozentpunkten
consistency = (df[models].max(axis=1) - df[models].min(axis=1)) <= threshold
print(f"## Prozentsatz der konsistenten Vorhersagen: {consistency.mean()*100:.2f}%")
print("\n")

# 4. Verteilung der Vorhersagen
plt.figure(figsize=(12, 6))
df[models].hist(bins=20)
plt.title('## Verteilung der Vorhersagen für jedes Modell')
plt.savefig('prediction_distribution.png')
plt.close()

# 5. Ausreißer und interessante Fälle
def find_interesting_cases(df, models):
    cases = []
    for index, row in df.iterrows():
        if row[models].max() - row[models].min() > 90:  # Große Diskrepanz
            cases.append((index, row['Filename'], row[models].to_dict()))
    return cases

interesting_cases = find_interesting_cases(df, models)
print("## Interessante Fälle (große Diskrepanz zwischen Modellen):")
for case in interesting_cases[:5]:  # Zeige die ersten 5 Fälle
    print(f"### Dokument: {case[1]}")
    for model, score in case[2].items():
        print(f"  {model:<12}\t{score:6.2f}")
    print()

# 6. Modellperformanz-Übersicht
model_performance = pd.DataFrame({
    'Mean': df[models].mean(),
    'Median': df[models].median(),
    'Std': df[models].std(),
    'Min': df[models].min(),
    'Max': df[models].max()

})


# Modellnamen
models = ['r-base', 'ms-deberta', 'distilb', 'r-large', 'albert']


# Prozentwerte in einem Array
thresholds = [50, 60, 70, 80, 90]

# Funktion zur Berechnung der Prozentsätze
def calculate_percentages(df, models, threshold):
    return [round((df[model] > threshold).mean() * 100, 2) for model in models]

# Erstellen der Tabelle
data = []
for threshold in thresholds:
    row = [f">{threshold}%"] + calculate_percentages(df, models, threshold)
    data.append(row)

# Tabelle erstellen und ausgeben
headers = ["Threshold"] + models
print("## Prozentsatz der Dokumente über Schwellenwerten:")

# Kopfzeile der Tabelle
print(f"{'Threshold':<12}" + "".join(f"{model:>12}" for model in models))
print("-" * (12 + 12 * len(models)))

# Datenzeilen
for row in data:
    print(f"{row[0]:<12}" + "".join(f"{value:12.2f}" for value in row[1:]))

print()


# Histogramme erstellen
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, model in enumerate(models):
    axes[i].hist(df[model], bins=50)
    axes[i].set_title(model)
    axes[i].set_xlabel('Konfidenz (%)')
    axes[i].set_ylabel('Anzahl der Dokumente')

plt.tight_layout()
plt.savefig('model_distributions.png')
plt.close()

# Detaillierte Statistiken
# print(df[models].describe())


# 4. Modellperformanz-Übersicht (Prozente, gerundet)

print("## Modellperformanz-Übersicht:")
performance_summary = df[models].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
print(performance_summary)
