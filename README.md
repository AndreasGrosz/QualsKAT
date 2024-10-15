# Authorship Attribution and QKAT CUDA Project

## Overview
This project combines authorship attribution techniques with CUDA-accelerated computations and model loading. It aims to determine the author of a given text using both generative (n-gram language model) and discriminative (sequence classifier) approaches, while leveraging CUDA for performance optimization.

## Project Components

### 1. Authorship Attribution
- **Generative Classifier**:
  - Uses NLTK's LM package for n-gram language models
  - Experiments with different smoothing techniques and backoff strategies
  - Handles out-of-vocabulary words
  - Generates samples and reports perplexity scores for each author

- **Discriminative Classifier**:
  - Utilizes Huggingface for sequence classification
  - Prepares data, creates train and test dataloaders
  - Trains the classifier using Huggingface Trainer class

### 2. CUDA and Model Loading
- Implements CUDA-accelerated computations for improved performance
- Manages model loading and processing

## Environment Setup

This project uses a local conda environment for managing dependencies.

1. Ensure you have Anaconda or Miniconda installed.

2. Clone the repository:
   ```
   git clone [your-repository-url]
   cd [your-project-directory]
   ```

3. Create and activate the local conda environment:
   ```
   conda create --prefix ./env python=3.10
   conda activate ./env
   ```

4. Install the required packages:
   ```
   conda env update --prefix ./env --file environment.yml --prune
   ```

5. Verify the installation:
   ```
   conda list
   ```

## Running the Project

### Authorship Attribution Component
```
python main.py --checkthis --quick

```

## Project Structure

- `main.py`: Main script for authorship attribution
- `file_utils.py`: Utility functions for file operations
- `model_utils.py`: Functions for model loading and processing
- `data_processing.py`: Data processing and preparation functions
- `analysis_utils.py`: Analysis utility functions
- `env/`: Local conda environment (do not edit directly)
- `models/`: Directory containing pre-trained models
- `CheckThis/`: Directory for files to be analyzed
- `output/`: Directory for output files

## Data Preparation
Ensure source files containing excerpts from various authors are available in the repository. Use UTF-8 encoding for consistency.

## Git and Environment Management

- The local conda environment (`./env`) is included in version control, excluding large binary files.
- After making changes to the environment, update `environment.yml`:
  ```
  conda env export --from-history > environment.yml
  ```

## Troubleshooting

If you encounter CUDA-related issues:
1. Ensure your CUDA drivers are up to date.
2. Verify that the installed PyTorch version is compatible with your CUDA version.
3. Check the `models` directory to ensure all required model files are present.

## Contributing

[Add any guidelines for contributing to your project]

## License

[Specify the license under which your project is released]
