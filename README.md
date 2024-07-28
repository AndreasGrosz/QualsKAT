## Authorship Attribution Project

### Overview
This project focuses on authorship attribution, aiming to determine the author of a given text. Two approaches, generative (n-gram language model) and discriminative (sequence classifier), are explored and compared in terms of their effectiveness in this task.

### Setup
I have compiled source files containing excerpts from works by various authors: Jane Austen, Charles Dickens, Leo Tolstoy, and Oscar Wilde. These files are available in the repo. Ensure to decide on the encoding type to use, preferably UTF-8.

### Project Details
- **Generative Classifier**:
  - Utilize NLTK's LM package to build n-gram language models, experimenting with different smoothing techniques and backoff strategies as outlined in Section 3.5 of Jurafsky and Martin.
  - Implement methods to handle out-of-vocabulary words during runtime.
  - Generate samples for each author using the trained language models and report the perplexity scores.
  - Optional: Implement n-gram language models from scratch using Numpy or PyTorch for bonus points.

- **Discriminative Classifier**:
  - Employ Huggingface to create a sequence classification model with k labels corresponding to the number of authors.
  - Prepare data, create train and test dataloaders, and train the classifier using the Huggingface Trainer class.

### Instructions
- Run the program `classifier.py` with the following command-line setups:
  ```
  python3 classifier.py authorlist -approach [generative|discriminative]
  python3 classifier.pt authorlist -approach [generative|discriminative] -test testfile
  ```
- When running without the `-test` flag, the program automatically extracts a development set, trains the model, and prints the results.
- When running with the `-test` flag, the program uses the entirety of data to train a language model and outputs classification results for each line in the given test file.
