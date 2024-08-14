import os
from PyPDF2 import PdfReader, PdfWriter

# Pfad zum Ordner mit den PDF-Dateien
pdf_folder_path = '.'

# Erhalte die Liste der PDF-Dateien (unabhängig von Groß-/Kleinschreibung)
pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]
print(f"Gefilterte PDF-Dateien: {pdf_files}")

# Erstelle Ordner und zerlege PDF-Dateien
for pdf_file in pdf_files:
    print(f"Beginne Verarbeitung von {pdf_file}...")
    try:
        folder_name = os.path.splitext(pdf_file)[0]
        folder_path = os.path.join(pdf_folder_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        pdf_file_path = os.path.join(pdf_folder_path, pdf_file)
        reader = PdfReader(pdf_file_path)

        for page_num in range(len(reader.pages)):
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])

            output_filename = os.path.join(folder_path, f'{folder_name}_page_{page_num + 1:04}.pdf')
            with open(output_filename, 'wb') as output_pdf:
                writer.write(output_pdf)

        print(f"Zerlegung von {pdf_file} abgeschlossen und Seiten in {folder_name} gespeichert.")

    except Exception as e:
        print(f"Ein Fehler ist bei der Verarbeitung von {pdf_file} aufgetreten: {e}")

print('Alle PDF-Dateien wurden erfolgreich zerlegt.')

# Nach dem Splitten: Konvertiere die gesplitteten PDFs in TXT
for root, dirs, files in os.walk(pdf_folder_path):
    for file in files:
        if file.lower().endswith('.pdf'):
            pdf_file_path = os.path.join(root, file)
            txt_file_path = os.path.splitext(pdf_file_path)[0] + '.txt'

            print(f"Konvertiere {pdf_file_path} nach {txt_file_path}...")

            # Lese die PDF und extrahiere den Text
            with open(pdf_file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()

            # Schreibe den extrahierten Text in eine TXT-Datei
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

            print(f"Konvertierung abgeschlossen: {txt_file_path}")

print("Alle PDF-Dateien wurden erfolgreich in TXT-Dateien konvertiert.")

