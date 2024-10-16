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

# Daten laden
df = pd.read_csv('output/HCOBs-analysis_results.csv')

# Modellnamen
models = ['r-base', 'ms-deberta', 'distilb', 'r-large', 'albert']

# 1. Deskriptive Statistik
print("Deskriptive Statistik:")
print(df[models].describe())
print("\n")

# 2. Korrelationsanalyse
corr = df[models].corr()
print("Korrelationsmatrix:")
print(corr)
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
print(f"Prozentsatz der konsistenten Vorhersagen: {consistency.mean()*100:.2f}%")
print("\n")

# 4. Verteilung der Vorhersagen
plt.figure(figsize=(12, 6))
df[models].hist(bins=20)
plt.title('Verteilung der Vorhersagen für jedes Modell')
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
print("Interessante Fälle (große Diskrepanz zwischen Modellen):")
for case in interesting_cases[:5]:  # Zeige die ersten 5 Fälle
    print(f"Dokument: {case[1]}")
    for model, score in case[2].items():
        print(f"  {model}: {score:.2f}")
    print()

# 6. Modellperformanz-Übersicht
model_performance = pd.DataFrame({
    'Mean': df[models].mean(),
    'Median': df[models].median(),
    'Std': df[models].std(),
    'Min': df[models].min(),
    'Max': df[models].max()
})
print("Modellperformanz-Übersicht:")
print(model_performance)

