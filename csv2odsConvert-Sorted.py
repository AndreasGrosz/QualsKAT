import pandas as pd
import re
import os
from odf.opendocument import OpenDocumentSpreadsheet
from odf.style import Style, TextProperties, TableColumnProperties
from odf.text import P
from odf.table import Table, TableColumn, TableRow, TableCell
from odf.text import A as TextA
from odf.number import NumberStyle, Number, ScientificNumber

def convert_filename(filename):
    if pd.isna(filename):
        return ''
    filename = str(filename)
    match = re.search(r'Seite_(\d+)', filename)
    if match:
        page_number = match.group(1).zfill(4)
        return filename.replace(f'Seite_{match.group(1)}', f'page_{page_number}')
    return filename

def create_hyperlink(filename):
    if pd.isna(filename):
        return ''
    filename = str(filename)
    full_path = os.path.join(os.getcwd(), filename)
    return f"file:///{full_path}"

def extract_band(filename):
    if pd.isna(filename):
        return ''
    filename = str(filename)
    match = re.search(r'(.+?)(?:_Seite_|_page)', filename)
    return match.group(1) if match else ''

# Lesen der CSV-Datei
df = pd.read_csv('ergebnisse.csv')

# Datentypen festlegen
numeric_columns = ['r-base', 'ms-deberta', 'distilb', 'r-large', 'albert', 'Mittelwert']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Neue Spalten hinzufügen
df['Sorted'] = df['Filename'].apply(convert_filename)
df['Hyperlink'] = df['Filename'].apply(create_hyperlink)
df['Band'] = df['Filename'].apply(extract_band)

# Sortieren
df = df.sort_values('Sorted')

# Erstellen des ODS-Dokuments
doc = OpenDocumentSpreadsheet()

# Stile
hyperlink_style = Style(name="Hyperlink", family="text")
hyperlink_style.addElement(TextProperties(color="#0000FF", textunderlinestyle="solid", textunderlinewidth="auto"))
doc.styles.addElement(hyperlink_style)

number_style = NumberStyle(name="Number")
number_style.addElement(Number(decimalplaces=1, minintegerdigits=1))
doc.styles.addElement(number_style)

# Tabelle erstellen
table = Table(name="Sheet1")

# Spalten definieren
for _ in range(len(df.columns)):
    table.addElement(TableColumn())

# Header-Zeile hinzufügen
header_row = TableRow()
for column in df.columns:
    cell = TableCell()
    cell.addElement(P(text=str(column)))
    header_row.addElement(cell)
table.addElement(header_row)

# Daten-Zeilen hinzufügen
for _, row in df.iterrows():
    table_row = TableRow()
    for column, value in row.items():
        cell = TableCell()
        if column == 'Hyperlink' and not pd.isna(value):
            p = P()
            link = TextA(href=value, text=str(row['Filename']), stylename=hyperlink_style)
            p.addElement(link)
            cell.addElement(p)
        elif column in numeric_columns:
            cell.setAttrNS('urn:oasis:names:tc:opendocument:xmlns:office:1.0', 'value-type', 'float')
            cell.setAttrNS('urn:oasis:names:tc:opendocument:xmlns:office:1.0', 'value', str(value))
            cell.addElement(P(text=f"{value:.1f}"))
        else:
            cell.addElement(P(text=str(value)))
        table_row.addElement(cell)
    table.addElement(table_row)

# Tabelle zum Dokument hinzufügen
doc.spreadsheet.addElement(table)

# Speichern des Dokuments
doc.save("ergebnisse.ods")
print("Conversion completed. The file 'ergebnisse.ods' has been created.")
print(f"Current working directory: {os.getcwd()}")
print("Please ensure that the text files are located in this directory.")
