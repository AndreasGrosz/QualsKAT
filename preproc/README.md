# OCR Error Detection Script

## Purpose
Analyzes text files for potential OCR errors by identifying unknown words and word fragments. Helps determine if OCR quality is sufficient for reliable author identification.

## Requirements
- Python 3.x
- NLTK library (automatically downloads required data)
- config.txt file with paths and thresholds
- ScnWorte.txt with domain-specific vocabulary

## Configuration (config.txt)
```ini
[Paths]
check_this = CheckThis/
output = output/

[OCR_Error_Evaluation]
unknown_words_threshold = 5
unknown_words_weight = 60
word_fragments_threshold = 5
word_fragments_weight = 40
min_word_length = 3
```

## Usage
### Debug Mode (Single File)
```bash
python OCR_Error_eval.py --debug
```
Shows colored text output:
- Green: Unknown words
- Yellow: Word fragments
- Plus detailed statistics

### Batch Mode
```bash
python OCR_Error_eval.py
```
Processes all .txt files in input directory, creates CSV output.

## Output CSV Structure
Fields in ocr_evaluation_results.csv:
- Dateiname: Source file name
- Unbekannte_Wörter: Percentage of unknown words
- Wortfragmente: Percentage of fragments
- Wortanzahl: Total word count
- Entscheidung: "Geeignet"/"Ungeeignet" based on thresholds
- Unbekannte_Wörter_Anzahl: Count of unknown words
- Wortfragmente_Anzahl: Count of fragments
- Dateigröße_Bytes: File size

## Decision Logic
Text is marked "Ungeeignet" if:
- Unknown words > 8% OR
- Word fragments > 12%

## Notes
- Handles scientology-specific terms via ScnWorte.txt
- Uses both US and GB English dictionaries
- Preserves line numbers in debug display
- Ignores OCR artifacts and punctuation
