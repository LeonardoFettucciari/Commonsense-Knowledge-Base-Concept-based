# Commonsense QA Concept Extraction and Statement Generation

This project processes the Commonsense QA dataset by extracting key concepts using Named Entity Recognition (NER) and generates commonsense statements using a Gemini language model.

## Features

- Loads and preprocesses Commonsense QA validation data
- Uses an NER pipeline to extract concept words
- Retrieves WordNet definitions for concepts
- Generates commonsense statements using a Gemini language model
- Saves outputs in a structured TSV file

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install datasets nltk tqdm google-generativeai babelscape
```

Additionally, download NLTK WordNet data:

```python
import nltk
nltk.download('wordnet')
```

## Usage

1. **Load and preprocess data**: The script loads 100 shuffled validation samples from the Commonsense QA dataset.
2. **Run NER**: It extracts relevant concepts using `Babelscape/cner-base`.
3. **Retrieve WordNet Definitions**: Extracted concepts are matched with their WordNet definitions.
4. **Generate Commonsense Statements**: A Gemini language model generates 10 statements per concept.
5. **Save Output**: Results are stored in `outputs/gemini-pro.tsv`.

Run the script:

```bash
python script.py
```

## Configuration

### Model Configuration

Modify the Gemini model settings in the script:

```python
api_key = "your_api_key_here"  
model_name = "gemini-pro"  
generation_config = {"temperature": 0.0, "max_output_tokens": 8192}
```

Replace `your_api_key_here` with your actual API key.

## Output Format

The results are saved in a TSV file with the following columns:

- `id`: Question ID
- `question`: The question text
- `choices`: The available answer choices
- `gold_truth`: The correct answer
- `statements`: Generated commonsense statements

## Acknowledgments

- [Commonsense QA Dataset](https://huggingface.co/datasets/tau/commonsense_qa)
- [Babelscape NER Model](https://huggingface.co/Babelscape/cner-base)
- [Google Gemini API](https://ai.google.dev/)

