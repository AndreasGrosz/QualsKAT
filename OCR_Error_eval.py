import os
import csv
import nltk
from nltk.corpus import words as nltk_words
from nltk.corpus import brown
from nltk.stem import PorterStemmer
import configparser
import argparse
from collections import Counter
import nltk
import string
import re  # Fügen Sie diesen Import


FILENAME = "610202 — HCO Bulletin — Command Sheet - Prehavingness Scale  [B001-009].txt"

# Download NLTK words if not already present
nltk.download('words', quiet=True)
nltk.download('brown', quiet=True)

ps = PorterStemmer()

# Definiere Worttrenner (alle Satzzeichen außer '/' und ''')
WORD_SEPARATORS = string.punctuation.replace('/', '').replace("'", "") + ' \t\n\r\v\f'

def load_english_words():
    us_words = set(word.lower() for word in nltk_words.words())
    gb_words = set(word.lower() for word in brown.words())
    return us_words, gb_words

us_words = set(word.lower() for word in nltk_words.words())
gb_words = set(word.lower() for word in brown.words())

# Funktion zum Laden von Wörterbüchern für andere Sprachen
def load_language_words(language):
    # Hier können Sie Logik zum Laden von Wörterbüchern für andere Sprachen hinzufügen
    # Beispiel: return set(word.lower() for word in open(f'{language}_words.txt', 'r').read().split())
    return set()

# Wörterbücher für andere Sprachen (Platzhalter)
de_words = load_language_words('de')
ru_words = load_language_words('ru')
fr_words = load_language_words('fr')
it_words = load_language_words('it')
es_words = load_language_words('es')


def display_colored_text(file_name, scn_words, english_words, config):
    ocr_config = config['OCR_Error_Evaluation']
    min_word_length = int(ocr_config['min_word_length'])

    file_path = os.path.join('CheckThis', file_name)

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenisiere den Text, aber behalte Zeilenumbrüche und Satzzeichen bei
    tokens = re.findall(r'\S+|\n|\s+|[^\w\s]', text)

    colored_text = ""
    for token in tokens:
        if token.isspace() or token == '\n' or token in string.punctuation:
            colored_text += token
        elif not is_word_known(token, scn_words, english_words, min_word_length):
            colored_text += f"\033[92m{token}\033[0m"  # Grün für unbekannte Wörter
        elif len(token.strip(WORD_SEPARATORS)) < min_word_length:
            colored_text += f"\033[93m{token}\033[0m"  # Gelb für Fragmente
        else:
            colored_text += token

    print(f"\nColored text for file: {file_name}\n")
    print(colored_text)


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


def is_single_letter_or_number(word):
    return len(word) == 1 and (word.isalpha() or word.isdigit())

def is_word_known(word, scn_words, english_words, min_word_length):
    original_word = word
    word = word.strip(WORD_SEPARATORS).lower()

    # Überprüfe die Stammform des Wortes
    stem = ps.stem(word)

    # Wenn das Wort kürzer als die Mindestlänge ist, markiere es als unbekannt
    if len(word) < min_word_length:
        return False

    # Überprüfe zuerst US-Englisch
    if word in us_words or stem in us_words:
        return True

    # , dann GB-Englisch als Fallback
    if word in gb_words or stem in gb_words:
        return True

# Überprüfe, ob das Wort im Scientology-Wörterbuch ist
    if original_word in scn_words or word in scn_words or stem in scn_words:
        return True

    # Überprüfe dann auf Datumsangaben, Zahlen und Kontraktionen
    if is_valid_date_or_number(original_word) or is_contraction(original_word):
        return True

    # Überprüfe auf zusammengesetzte Wörter
    if '-' in word:
        parts = word.split('-')
        if all(part in scn_words or part in us_words or part in gb_words for part in parts):
            return True

    return False

def load_scn_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f)


def debug_file(file_name, scn_words, english_words, config):
    ocr_config = config['OCR_Error_Evaluation']
    min_word_length = int(ocr_config['min_word_length'])
    unknown_words_threshold = float(ocr_config['unknown_words_threshold'])
    word_fragments_threshold = float(ocr_config['word_fragments_threshold'])
    unknown_weight = float(ocr_config['unknown_words_weight'])
    fragment_weight = float(ocr_config['word_fragments_weight'])

    file_path = os.path.join('CheckThis', file_name)

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    words_in_text = tokenize_text(text)
    total_words = len(words_in_text)

    unknown_words = []
    fragments = []

    print("\nWord-by-word analysis:")

    for word in words_in_text:
        issues = []
        if not is_word_known(word, scn_words, english_words, min_word_length):
            unknown_words.append(word)
            issues.append("Unknown")
        if len(word.strip(WORD_SEPARATORS)) < min_word_length:
            fragments.append(word)
            issues.append("Fragment")

        if issues:
            print(f"- {word}: {', '.join(issues)}")

    unknown_percentage = (len(unknown_words) / total_words) * 100
    fragment_percentage = (len(fragments) / total_words) * 100

    display_colored_text(file_name, scn_words, english_words, config)

    print()
    print(f"Analyzing file: {file_name}")
    print(f"Total words: {total_words}")
    print()
    print(f"\nUnknown words: {len(unknown_words)} ({unknown_percentage:.2f}%)")
    print(f"Word fragments: {len(fragments)} ({fragment_percentage:.2f}%)")

    # Calculate weighted score and decision
    score = 100 - ((unknown_percentage * unknown_weight / 100) +
                   (fragment_percentage * fragment_weight / 100))

    if unknown_percentage <= unknown_words_threshold and fragment_percentage <= word_fragments_threshold:
        decision = "Geeignet"
    else:
        decision = "Ungeeignet"

    print(f"\nGewichtete Auswertung:")
    print(f"Gesamtscore: {score:.2f}%")
    print(f"Entscheidung: {decision}")
    print(f"Schwellenwerte: Unbekannte Wörter {unknown_words_threshold}%, Wortfragmente {word_fragments_threshold}%")

    return {
        "Dateiname": os.path.basename(file_path),
        "Gesamtscore": f"{score:.2f}%",
        "Unbekannte_Wörter": f"{unknown_percentage:.2f}%",
        "Wortfragmente": f"{fragment_percentage:.2f}%",
        "Wortanzahl": total_words,
        "Entscheidung": decision,
        "Unbekannte_Wörter_Anzahl": len(unknown_words),
        "Wortfragmente_Anzahl": len(fragments),
        "Unbekannte_Wörter_Schwellenwert": f"{unknown_words_threshold}%",
        "Wortfragmente_Schwellenwert": f"{word_fragments_threshold}%",
        "Dateigröße_Bytes": os.path.getsize(file_path)
    }

def load_config():
    config = configparser.ConfigParser()
    config.read('config.txt')
    return config

def load_scn_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f)

def is_word_fragment(word, min_length):
    return len(word) < min_length

def analyze_file(file_path, scn_words, english_words, config):
    ocr_config = config['OCR_Error_Evaluation']
    min_word_length = int(ocr_config['min_word_length'])
    unknown_words_threshold = float(ocr_config['unknown_words_threshold'])
    word_fragments_threshold = float(ocr_config['word_fragments_threshold'])
    unknown_weight = float(ocr_config['unknown_words_weight'])
    fragment_weight = float(ocr_config['word_fragments_weight'])

    try:
        file_size = os.path.getsize(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[6:-6]  # Ignore first 6 and last 6 lines

        text = ' '.join(lines).lower()
        words_in_text = text.split()
        total_words = len(words_in_text)

        if total_words == 0:
            print(f"Warning: No words found in {file_path}")
            return None

        unknown_words = [w for w in words_in_text if not is_word_known(w, scn_words, english_words, min_word_length)]
        fragments = [w for w in words_in_text if is_word_fragment(w, min_word_length)]

        unknown_percentage = (len(unknown_words) / total_words) * 100
        fragment_percentage = (len(fragments) / total_words) * 100

        if unknown_percentage <= unknown_words_threshold and fragment_percentage <= word_fragments_threshold:
            decision = "Geeignet"
        else:
            decision = "Ungeeignet"

        score = 100 - ((unknown_percentage * unknown_weight / 100) +
                    (fragment_percentage * fragment_weight / 100))

        return {
            "Dateiname": os.path.basename(file_path),
            "Gesamtscore": f"{score:.2f}%",
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

    scn_words = load_scn_words(args.scn_words)
    english_words = set(word.lower() for word in nltk_words.words())

    input_dir = config['Paths']['check_this']
    output_dir = config['Paths']['output']

    results = []  # Initialize results here

    if args.debug:
        debug_result = debug_file(filename, scn_words, english_words, config)
        if debug_result:
            print()
           # results.append(debug_result)
    else:
        print("Normal mode: Processing all files")
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_dir, filename)
                try:
                    result = analyze_file(file_path, scn_words, english_words, config)
                    if result:
                        results.append(result)

                        print(f"\nAnalyse für {result['Dateiname']}:")
                        print(f"Gesamtscore: {result['Gesamtscore']}")
                        print(f"Unbekannte Wörter: {result['Unbekannte_Wörter']} ({result['Unbekannte_Wörter_Anzahl']} von {result['Wortanzahl']})")
                        print(f"Wortfragmente: {result['Wortfragmente']} ({result['Wortfragmente_Anzahl']} von {result['Wortanzahl']})")
                        print(f"Entscheidung: {result['Entscheidung']}")
                        print(f"Dateigröße: {result['Dateigröße_Bytes']} Bytes")
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {filename}: {str(e)}")

    if results:  # Only write to CSV if we have results
        output_file = os.path.join(output_dir, 'ocr_evaluation_results.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Dateiname", "Gesamtscore", "Unbekannte_Wörter", "Wortfragmente", "Wortanzahl", "Entscheidung",
                          "Unbekannte_Wörter_Anzahl", "Wortfragmente_Anzahl", "Unbekannte_Wörter_Schwellenwert",
                          "Wortfragmente_Schwellenwert", "Dateigröße_Bytes"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Ergebnisse wurden in {output_file} gespeichert.")
    else:
        print("Keine Ergebnisse zum Speichern.")

if __name__ == "__main__":
    main()
