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

FILENAME = "600129 — HCO Bulletin — Congresses  [B036-023].txt"

# Download NLTK words if not already present
nltk.download('words', quiet=True)
nltk.download('brown', quiet=True)

ps = PorterStemmer()

# Definiere Worttrenner (alle Satzzeichen außer '/' und ''')
WORD_SEPARATORS = string.punctuation.replace('/', '').replace("'", "") + ' \t\n\r\v\f'

def load_scn_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f)

def tokenize_text(text):
    return re.findall(r"\b[\w/'.-]+\b", text)

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

def display_colored_text(file_name, EN_SCN_words, EN_US_words, EN_GB_words, config):
    ocr_config = config['OCR_Error_Evaluation']
    min_word_length = int(ocr_config['min_word_length'])

    file_path = os.path.join('CheckThis', file_name)

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = re.findall(r'\S+|\n|\s+|[^\w\s]', text)

    colored_text = ""
    for token in tokens:
        if token.isspace() or token == '\n' or token in string.punctuation:
            colored_text += token
        elif is_word_known(token, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length):
            colored_text += token
        elif len(token.strip(WORD_SEPARATORS)) < min_word_length:
            colored_text += f"\033[93m{token}\033[0m"  # Gelb für Fragmente
        else:
            colored_text += f"\033[92m{token}\033[0m"  # Grün für unbekannte Wörter

    print(f"\nColored text for file: {file_name}")
    """print("\033[92mUnbekannte Wörter\033[0m und \033[93mWortfragmente\033[0m")"""
    print(colored_text)

def debug_file(file_name, EN_SCN_words, EN_US_words, EN_GB_words, config):
    ocr_config = config['OCR_Error_Evaluation']
    min_word_length = int(ocr_config['min_word_length'])
    unknown_words_threshold = float(ocr_config['unknown_words_threshold'])
    word_fragments_threshold = float(ocr_config['word_fragments_threshold'])

    file_path = os.path.join('CheckThis', file_name)

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    file_size = os.path.getsize(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    words_in_text = tokenize_text(text)
    total_words = len(words_in_text)

    unknown_words = []
    fragments = []

    for word in words_in_text:
        if not is_word_known(word, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length):
            if len(word.strip(WORD_SEPARATORS)) < min_word_length:
                fragments.append(word)
            else:
                unknown_words.append(word)

    unknown_percentage = (len(unknown_words) / total_words) * 100
    fragment_percentage = (len(fragments) / total_words) * 100

    unknown_ok = unknown_percentage <= unknown_words_threshold
    fragments_ok = fragment_percentage <= word_fragments_threshold
    decision = "Geeignet" if (unknown_ok and fragments_ok) else "Ungeeignet"

    # Farbcodes
    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"

    display_colored_text(file_name, EN_SCN_words, EN_US_words, EN_GB_words, config)

    # Statistik-Ausgabe
    print("\n============================")
    print(f"Analyzing file: \n{file_name}")
    print(f"Total words:            {total_words}")
    print(f"\033[92mUnbekannte Wörter\033[0m       {len(unknown_words)} = {unknown_percentage:.1f}%, " +
          (f"{GREEN}unter Grenzwert{RESET}" if unknown_ok else f"{RED}über Grenzwert{RESET}") +
          f" {unknown_words_threshold}%")
    print(f"\033[93mWortfragmente\033[0m           {len(fragments)} = {fragment_percentage:.1f}%, " +
          (f"{GREEN}unter Grenzwert{RESET}" if fragments_ok else f"{RED}über Grenzwert{RESET}") +
          f" {word_fragments_threshold}%")
    print(f"Entscheidung:           " +
          (f"{GREEN}Geeignet{RESET}" if decision == "Geeignet" else f"{RED}Ungeeignet{RESET}"))

    return {
        "Dateiname": os.path.basename(file_path),
        "Unbekannte_Wörter": f"{unknown_percentage:.2f}%",
        "Wortfragmente": f"{fragment_percentage:.2f}%",
        "Wortanzahl": total_words,
        "Entscheidung": decision,
        "Unbekannte_Wörter_Anzahl": len(unknown_words),
        "Wortfragmente_Anzahl": len(fragments),
        "Unbekannte_Wörter_Schwellenwert": f"{unknown_words_threshold}%",
        "Wortfragmente_Schwellenwert": f"{word_fragments_threshold}%",
        "Dateigröße_Bytes": file_size
    }

def analyze_file(file_path, EN_SCN_words, EN_US_words, EN_GB_words, config):
    ocr_config = config['OCR_Error_Evaluation']
    min_word_length = int(ocr_config['min_word_length'])
    unknown_words_threshold = float(ocr_config['unknown_words_threshold'])
    word_fragments_threshold = float(ocr_config['word_fragments_threshold'])

    try:
        file_size = os.path.getsize(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()  # Lesen des kompletten Textes ohne Zeilen zu ignorieren

        words_in_text = tokenize_text(text)
        total_words = len(words_in_text)

        if total_words == 0:
            print(f"Warning: No words found in {file_path}")
            return None

        unknown_words = []
        fragments = []

        for word in words_in_text:
            if not is_word_known(word, EN_SCN_words, EN_US_words, EN_GB_words, min_word_length):
                if len(word.strip(WORD_SEPARATORS)) < min_word_length:
                    fragments.append(word)
                else:
                    unknown_words.append(word)

        unknown_percentage = (len(unknown_words) / total_words) * 100
        fragment_percentage = (len(fragments) / total_words) * 100

        unknown_ok = unknown_percentage <= unknown_words_threshold
        fragments_ok = fragment_percentage <= word_fragments_threshold
        decision = "Geeignet" if (unknown_ok and fragments_ok) else "Ungeeignet"

        # Farbcodes
        GREEN = "\033[32m"
        RED = "\033[31m"
        RESET = "\033[0m"

        print(f"\nAnalyse für {os.path.basename(file_path)}:")
        print(f"Total words:            {total_words}")
        print(f"\033[92mUnbekannte Wörter\033[0m       {len(unknown_words)} = {unknown_percentage:.1f}%, " +
              (f"{GREEN}unter Grenzwert{RESET}" if unknown_ok else f"{RED}über Grenzwert{RESET}") +
              f" {unknown_words_threshold}%")
        print(f"\033[93mWortfragmente\033[0m           {len(fragments)} = {fragment_percentage:.1f}%, " +
              (f"{GREEN}unter Grenzwert{RESET}" if fragments_ok else f"{RED}über Grenzwert{RESET}") +
              f" {word_fragments_threshold}%")
        print(f"Entscheidung:           " +
              (f"{GREEN}Geeignet{RESET}" if decision == "Geeignet" else f"{RED}Ungeeignet{RESET}"))

        return {
            "Dateiname": os.path.basename(file_path),
            "Unbekannte_Wörter": f"{unknown_percentage:.2f}%",
            "Wortfragmente": f"{fragment_percentage:.2f}%",
            "Wortanzahl": total_words,
            "Entscheidung": decision,
            "Unbekannte_Wörter_Anzahl": len(unknown_words),
            "Wortfragmente_Anzahl": len(fragments),
            "Dateigröße_Bytes": file_size
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

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
        debug_result = debug_file(filename, EN_SCN_words, EN_US_words, EN_GB_words, config)
        if debug_result:
            results.append(debug_result)
    else:
        print("Normal mode: Processing all files")
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_dir, filename)
                try:
                    result = analyze_file(file_path, EN_SCN_words, EN_US_words, EN_GB_words, config)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {filename}: {str(e)}")

        if results:  # Only write to CSV if we have results
            output_file = os.path.join(output_dir, 'ocr_evaluation_results.csv')
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["Dateiname", "Unbekannte_Wörter", "Wortfragmente",
                            "Wortanzahl", "Entscheidung", "Unbekannte_Wörter_Anzahl",
                            "Wortfragmente_Anzahl", "Dateigröße_Bytes"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    # Entfernen der nicht benötigten Felder aus dem result Dictionary
                    result.pop("Unbekannte_Wörter_Schwellenwert", None)
                    result.pop("Wortfragmente_Schwellenwert", None)
                    writer.writerow(result)

if __name__ == "__main__":
    main()
