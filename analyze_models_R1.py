import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

OUTPUT_DIR = 'output'

if len(sys.argv) < 2:
    print("Bitte geben Sie den Dateinamen als Parameter an.")
    sys.exit(1)

input_file = sys.argv[1]
input_file_path = os.path.join(OUTPUT_DIR, input_file)
df = pd.read_csv(input_file_path)

# Modellnamen
models = [col for col in df.columns if col not in ['Dateiname', 'Dateigröße', 'Datum', 'Mittelwert']]

if 'Mittelwert' in df.columns:
    models.remove('Mittelwert')

# Überschrift
current_date = date.today().strftime("%y%m%d")
input_file_name = os.path.splitext(input_file)[0]
print(f"# {current_date} Analyse Models Ausgabe {input_file_name}.md")

# 1. Deskriptive Statistik
print("## Deskriptive Statistik:")
stats_columns = models + (['Mittelwert'] if 'Mittelwert' in df.columns else [])
print(df[stats_columns].describe().round(2))
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
threshold = 10
consistency = (df[models].max(axis=1) - df[models].min(axis=1)) <= threshold
print(f"## Prozentsatz der konsistenten Vorhersagen: {consistency.mean()*100:.2f}%")
print("\n")

# 4. Verteilung der Vorhersagen
plt.figure(figsize=(12, 6))
df[models + ['Mittelwert']].hist(bins=20)
plt.title('Verteilung der Vorhersagen für jedes Modell')
plt.savefig('prediction_distribution.png')
plt.close()

# 5. Ausreißer und interessante Fälle
def find_interesting_cases(df, models):
    cases = []
    for index, row in df.iterrows():
        if row[models].max() - row[models].min() > 90:
            cases.append((index, row['Dateiname'], row[models].to_dict()))
    return cases

interesting_cases = find_interesting_cases(df, models)
print("## Interessante Fälle (große Diskrepanz zwischen Modellen):")
for case in interesting_cases[:5]:
    print(f"### Dokument: {case[1]}")
    for model, score in case[2].items():
        print(f"  {model:<12}\t{score:6d}")
    print()

# 6. Modellperformanz-Übersicht
print("## Modellperformanz-Übersicht:")
performance_columns = models + (['Mittelwert'] if 'Mittelwert' in df.columns else [])
performance_summary = df[performance_columns].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
print(performance_summary)

# Schwellenwertanalyse
thresholds = [50, 60, 70, 80, 90]

print("## Prozentsatz der Dokumente über Schwellenwerten:")
header_columns = models + (['Mittelwert'] if 'Mittelwert' in df.columns else [])
print(f"{'Threshold':<12}" + "".join(f"{model:>12}" for model in header_columns))
print("-" * (12 + 12 * len(header_columns)))

for threshold in thresholds:
    row = [f">{threshold}%"] + [round((df[model] > threshold).mean() * 100, 2) for model in header_columns]
    print(f"{row[0]:<12}" + "".join(f"{value:12.2f}" for value in row[1:]))

# Histogramme
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, model in enumerate(header_columns):
    if i < len(axes):
        axes[i].hist(df[model], bins=50)
        axes[i].set_title(model)
        axes[i].set_xlabel('Konfidenz (%)')
        axes[i].set_ylabel('Anzahl der Dokumente')

plt.tight_layout()
plt.savefig('model_distributions.png')
plt.close()
