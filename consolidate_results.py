"""
    Konsolidiert die Ergebnisse mehrerer Modellvorhersagen in eine einzige CSV-Datei.

    Diese Funktion:
    1. Liest alle 'CheckThisResults_*.csv' Dateien aus dem konfigurierten Ausgabeordner.
    2. Kombiniert die Daten aller Modelle, wobei nur die LRH-Wahrscheinlichkeiten behalten werden.
    3. Überprüft, ob alle Modelle die gleichen Dateinamen verarbeitet haben.
    4. Sortiert die Ergebnisse nach Dateinamen (alphanumerisch).
    5. Erstellt eine Pivot-Tabelle mit Dateiname, Größe, Datum und LRH-Werten für jedes Modell.
    6. Speichert das Ergebnis in einer neuen CSV-Datei, deren Name den Zeitstempel der Analyse enthält.

    Parameter:
    config (ConfigParser): Konfigurationsobjekt mit Pfadangaben

    Die resultierende CSV-Datei enthält folgende Spalten:
    - Dateiname
    - Dateigröße
    - Datum der Analyse
    - LRH-Wahrscheinlichkeiten für jedes Modell
"""
import pandas as pd
import os
import re
import configparser
from datetime import datetime

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def consolidate_results(config):
    output_folder = config['Paths']['output']
    csv_files = [f for f in os.listdir(output_folder) if f.startswith("CheckThisResults_") and f.endswith(".csv")]

    all_data = []
    first_timestamp = None

    for csv_file in csv_files:
        model_name = csv_file.split("_")[1].split(".")[0]
        df = pd.read_csv(os.path.join(output_folder, csv_file))
        df['Model'] = model_name
        all_data.append(df)

        if first_timestamp is None and not df.empty:
            first_timestamp = df['Datum'].iloc[0]

    if not all_data:
        print("Keine Daten zum Konsolidieren gefunden.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Überprüfen, ob alle Dateinamen in allen Modellen identisch sind
    filenames = combined_df.groupby('Model')['Dateiname'].apply(set)
    if not all(filenames.iloc[0] == names for names in filenames):
        print("Warnung: Nicht alle Modelle haben identische Dateinamen verarbeitet.")

    # Sortieren und Pivot
    combined_df['Dateiname'] = combined_df['Dateiname'].astype(str)
    combined_df = combined_df.sort_values(by='Dateiname', key=lambda x: x.map(natural_sort_key))

    # Behalten Sie die Dateigröße und das Datum vom ersten Modell
    first_model = combined_df['Model'].iloc[0]
    file_info = combined_df[combined_df['Model'] == first_model][['Dateiname', 'Dateigröße', 'Datum']]

    pivot_df = combined_df.pivot(index='Dateiname', columns='Model', values='LRH')

    # Konvertieren zu Prozent und auf ganze Zahlen runden
    pivot_df = pivot_df.map(lambda x: int(x * 100) if pd.notnull(x) else x)

    # Fügen Sie Dateigröße und Datum wieder hinzu
    result_df = file_info.merge(pivot_df.reset_index(), on='Dateiname')

    # Berechnen Sie den Mittelwert
    models = [col for col in result_df.columns if col not in ['Dateiname', 'Dateigröße', 'Datum']]
    result_df['Mittelwert'] = result_df[models].mean(axis=1).astype(int)

    # Sortieren der Spalten
    columns = ['Dateiname', 'Dateigröße', 'Datum'] + sorted(models) + ['Mittelwert']
    result_df = result_df[columns]

    # Erstellen des Ausgabedateinamens
    if first_timestamp:
        timestamp = datetime.strptime(first_timestamp, "%d-%m-%y %H:%M")
        output_filename = f"{timestamp.strftime('%y%m%d-%Hh%M')}-Results.csv"
    else:
        output_filename = "ConsolidatedResults.csv"

    output_file = os.path.join(output_folder, output_filename)
    result_df.to_csv(output_file, index=False)
    print(f"Konsolidierte Ergebnisse wurden in {output_file} gespeichert.")

# Hauptausführung
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.txt')
    consolidate_results(config)
