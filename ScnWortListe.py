import os
import re
from striprtf.striprtf import rtf_to_text

def extract_lemmata(rtf_content):
    # Konvertiere RTF zu reinem Text
    plain_text = rtf_to_text(rtf_content)

    # Teile den Text in Absätze
    paragraphs = plain_text.split('\n\n')

    lemmata = set()
    for paragraph in paragraphs:
        # Finde das Lemma (Wörter bis zum ersten Doppelpunkt)
        match = re.match(r'^(.*?):', paragraph.strip())
        if match:
            lemma = match.group(1).strip()
            lemmata.add(lemma)

    return lemmata

def process_rtf_files(input_folder):
    all_lemmata = set()

    for filename in os.listdir(input_folder):
        if filename.endswith('.rtf'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                rtf_content = file.read()
                file_lemmata = extract_lemmata(rtf_content)
                all_lemmata.update(file_lemmata)

    return sorted(all_lemmata, key=str.lower)

# Hauptprogramm
input_folder = 'import'
output_file = 'ScnWorte.txt'

lemmata = process_rtf_files(input_folder)

with open(output_file, 'w', encoding='utf-8') as file:
    for lemma in lemmata:
        file.write(f"{lemma}\n")

print(f"{len(lemmata)} unique Lemmata wurden in {output_file} gespeichert.")
