import os
import csv
import nltk
from nltk.corpus import words as nltk_words
from nltk.corpus import brown
from nltk.stem import PorterStemmer
import configparser
import argparse
from collections import Counter
import string
import re

FILENAME = "670630 — HCO Bulletin — Evidences of an Aberrated Area  [B041-028].txt"

# Download NLTK words if not already present
nltk.download('words', quiet=True)
nltk.download('brown', quiet=True)

ps = PorterStemmer()

# Definiere Worttrenner (alle Satzzeichen außer '/' und ''')
WORD_SEPARATORS = string.punctuation.replace('/', '').replace("'", "") + ' \t\n\r\v\f'

def load_scn_words(file_path):
    """Lädt die Scientology-spezifischen Wörter aus einer Datei"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(word.strip().lower() for word in f)
    except Exception as e:
        print(f"Error loading Scientology word list from {file_path}: {str(e)}")
        return set()

def is_valid_date_or_number(word):
    return bool(re.match(r'^\d{1,2}([./-])\d{1,2}\1\d{2,4}$', word) or  # Dates like 28.9.80
                re.match(r'^\d+(/\d+)?$', word))  # Numbers like 1/2

def is_contraction(word):
    contractions = ["can't", "couldn't", "didn't", "don't", "isn't", "weren't", "you're"]
    return word.lower() in contractions

def is_word_known(word, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length):
    # Entferne Satzzeichen und konvertiere zu Kleinbuchstaben
    original_word = word
    word = word.strip(WORD_SEPARATORS).lower()

    # Sonderfälle für einbuchstabige Wörter
    if word in ["i", "a"]:
        return True

    # Bilde den Wortstamm
    stem = ps.stem(word)

    # Prüfe das Originalwort und den Stamm in den Wörterbüchern
    for w in [word, stem]:
        if w in EN_US_words:
            return True
        if w in EN_GB_words:
            return True
        if w in EN_SCN_words:
            return True

    # Prüfe auf Datumsangaben, Zahlen und Kontraktionen
    if is_valid_date_or_number(original_word) or is_contraction(original_word):
        return True

    # Prüfe auf zusammengesetzte Wörter
    if '-' in word:
        parts = word.split('-')
        if all(part in EN_US_words or part in EN_GB_words or part in EN_SCN_words for part in parts):
            return True

    return False

def analyze_word(token, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length):
    """Zentrale Funktion zur Wortanalyse"""
    if token.isspace() or token in string.punctuation:
        return "space"

    # Entferne Nummerierung am Zeilenanfang
    if re.match(r'^\s*\d+\.\s*$', token):
        return "number"

    if is_word_known(token, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length):
        return "known"
    elif len(token.strip(WORD_SEPARATORS)) < min_word_length:
        return "fragment"
    else:
        return "unknown"

def analyze_text(text, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length):
    """Zentrale Analysefunktion für einen Text"""
    # Erst die Zeilen verarbeiten
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        # Nummerierung am Zeilenanfang durch Leerzeichen ersetzen
        processed_line = re.sub(r'^\s*\d+\.\s*', ' ', line)
        processed_lines.append(processed_line)

    # Text wieder zusammenfügen und tokenisieren
    processed_text = '\n'.join(processed_lines)
    tokens = re.findall(r'\S+|\s+|[^\w\s]', processed_text)

    unknown_words = []
    fragments = []
    total_words = 0
    word_analysis = []

    for token in tokens:
        result = analyze_word(token, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length)
        word_analysis.append((token, result))

        if result in ["known", "unknown", "fragment"]:
            total_words += 1
            if result == "unknown":
                unknown_words.append(token)
            elif result == "fragment":
                fragments.append(token)

    return {
        "word_analysis": word_analysis,
        "unknown_words": unknown_words,
        "fragments": fragments,
        "total_words": total_words
    }

def display_colored_text(file_name, text, EN_SCN_words, EN_US_words, EN_GB_words, config):
    min_word_length = int(config['OCR_Error_Evaluation']['min_word_length'])
    tokens = re.findall(r'\S+|\s+|[^\w\s]', text)
    colored_text = ""

    for token in tokens:
        if re.match(r'^\s*\d+\.\s*$', token):  # Nummerierung
            colored_text += token
        elif token.isspace() or token in string.punctuation:
            colored_text += token
        elif is_word_known(token, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length):
            colored_text += token
        elif len(token.strip(WORD_SEPARATORS)) < min_word_length:
            colored_text += f"\033[93m{token}\033[0m"  # Gelb
        else:
            colored_text += f"\033[92m{token}\033[0m"  # Grün

    print(f"\nColored text for file: {file_name}")
    print(colored_text)

def generate_statistics(analysis_results, config):
    """Generiert die statistischen Daten für Debug und CSV"""
    total_words = analysis_results["total_words"]
    unknown_count = len(analysis_results["unknown_words"])
    fragment_count = len(analysis_results["fragments"])

    unknown_percentage = (unknown_count / total_words) * 100 if total_words > 0 else 0
    fragment_percentage = (fragment_count / total_words) * 100 if total_words > 0 else 0

    # Schwellenwerte aus der Konfiguration lesen
    unknown_words_threshold = float(config['OCR_Error_Evaluation']['unknown_words_threshold'])
    word_fragments_threshold = float(config['OCR_Error_Evaluation']['word_fragments_threshold'])

    unknown_ok = unknown_percentage <= unknown_words_threshold
    fragments_ok = fragment_percentage <= word_fragments_threshold
    decision = "Geeignet" if (unknown_ok and fragments_ok) else "Ungeeignet"

    return {
        "Wortanzahl": total_words,
        "Unbekannte_Wörter_Anzahl": unknown_count,
        "Wortfragmente_Anzahl": fragment_count,
        "Unbekannte_Wörter": f"{unknown_percentage:.2f}%",
        "Wortfragmente": f"{fragment_percentage:.2f}%",
        "Entscheidung": decision,
        "unknown_percentage": unknown_percentage,  # für Debug-Ausgabe
        "fragment_percentage": fragment_percentage,  # für Debug-Ausgabe
        "unknown_ok": unknown_ok,  # für Debug-Ausgabe
        "fragments_ok": fragments_ok  # für Debug-Ausgabe
    }

def process_file(file_path, EN_SCN_words, EN_US_words, EN_GB_words, config, debug=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Für die Statistik: Text ohne Nummerierung
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        processed_line = re.sub(r'^\s*\d+\.\s*', ' ', line)
        processed_lines.append(processed_line)
    processed_text = '\n'.join(processed_lines)

    # Analyse mit bereinigtem Text
    analysis_results = analyze_text(processed_text, EN_SCN_words, EN_US_words, EN_GB_words,
                                  int(config['OCR_Error_Evaluation']['min_word_length']))

    stats = generate_statistics(analysis_results, config)

    if debug:
        # Farbige Textanzeige mit Originaltext
        display_colored_text(os.path.basename(file_path), text, EN_SCN_words, EN_US_words, EN_GB_words, config)

        # Debug-Statistik ausgeben
        GREEN = "\033[32m"
        RED = "\033[31m"
        RESET = "\033[0m"

        print("\n============================")
        print(f"Analyzing file: \n{os.path.basename(file_path)}")
        print(f"Total words:            {stats['Wortanzahl']}")
        print(f"\033[92mUnbekannte Wörter\033[0m       {stats['Unbekannte_Wörter_Anzahl']} = {stats['unknown_percentage']:.1f}%, " +
              (f"{GREEN}unter Grenzwert{RESET}" if stats['unknown_ok'] else f"{RED}über Grenzwert{RESET}") +
              f" {config['OCR_Error_Evaluation']['unknown_words_threshold']}%")
        print(f"\033[93mWortfragmente\033[0m           {stats['Wortfragmente_Anzahl']} = {stats['fragment_percentage']:.1f}%, " +
              (f"{GREEN}unter Grenzwert{RESET}" if stats['fragments_ok'] else f"{RED}über Grenzwert{RESET}") +
              f" {config['OCR_Error_Evaluation']['word_fragments_threshold']}%")
        print(f"Entscheidung:           " +
              (f"{GREEN}Geeignet{RESET}" if stats['Entscheidung'] == 'Geeignet' else f"{RED}Ungeeignet{RESET}"))

    # Ergänze Dateiinformationen für CSV
    stats["Dateiname"] = os.path.basename(file_path)
    stats["Dateigröße_Bytes"] = os.path.getsize(file_path)

    # Entferne Debug-spezifische Felder für CSV
    stats.pop("unknown_percentage", None)
    stats.pop("fragment_percentage", None)
    stats.pop("unknown_ok", None)
    stats.pop("fragments_ok", None)

    return stats

def main():
    filename = FILENAME
    parser = argparse.ArgumentParser(description="OCR Error Detection")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--scn_words", default="preproc/ScnWortListe/ScnWorte.txt", help="Path to Scientology word list")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.txt')

    input_dir = config['Paths']['check_this']
    output_dir = config['Paths']['output']

    EN_SCN_words = load_scn_words(args.scn_words)
    EN_US_words = set(word.lower() for word in nltk_words.words())
    EN_GB_words = set(word.lower() for word in brown.words())

    results = []
    if args.debug:
        file_path = os.path.join('CheckThis', filename)
        result = process_file(file_path, EN_SCN_words, EN_US_words, EN_GB_words, config, debug=True)
        if result:
            results.append(result)
    else:
        print("Normal mode: Processing all files")
        files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        total_files = len(files)

        for idx, filename in enumerate(files, 1):
            file_path = os.path.join(input_dir, filename)
            try:
                result = process_file(file_path, EN_SCN_words, EN_US_words, EN_GB_words, config)
                if result:
                    results.append(result)
                # Fortschrittsanzeige (überschreibt die vorherige Zeile)
                print(f"\rProcessing file {idx}/{total_files}: {filename}", end="", flush=True)

            except Exception as e:
                print(f"\nFehler bei der Verarbeitung von {filename}: {str(e)}")

        print("\nVerarbeitung abgeschlossen.")  # Neue Zeile nach der Fortschrittsanzeige

        if results:  # Only write to CSV if we have results
            output_file = os.path.join(output_dir, 'ocr_evaluation_results.csv')
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["Dateiname", "Unbekannte_Wörter", "Wortfragmente",
                            "Wortanzahl", "Entscheidung", "Unbekannte_Wörter_Anzahl",
                            "Wortfragmente_Anzahl", "Dateigröße_Bytes"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result)

if __name__ == "__main__":
    main()
