<div align="center">
  <img src="assets/sapienzanlp.png" width="125">

# Retrieval-Augmented Generation for Commonsense Reasoning: An Empirical Investigation of Challenges and Limitations
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Library-HuggingFace-yellow.svg?logo=huggingface)](https://huggingface.co/)
[![ChatGPT](https://img.shields.io/badge/LLM-ChatGPT-1abc9c.svg?logo=openai)](https://chat.openai.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4.svg?logo=googlegemini)](https://ai.google/gemini/)
[![RAG](https://img.shields.io/badge/Method-RAG-orange.svg)](https://arxiv.org/abs/2005.11401)
[![Commonsense Reasoning](https://img.shields.io/badge/Task-Commonsense%20Reasoning-lightgrey.svg)](https://en.wikipedia.org/wiki/Commonsense_reasoning)
[![Sapienza](https://img.shields.io/badge/University-Sapienza-b31b1b.svg)](https://sapienzanlp.uniroma1.it/)

This repository provides the code and resources for investigating how retrieval-augmented generation (RAG) systems handle commonsense reasoning tasks. We focus on three key research questions:
How do design choices in commonsense knowledge base (KB) construction affect downstream performance?
What are the strengths and limitations of different retrieval strategies for commonsense knowledge?
To what extent can large language models (LLMs) effectively leverage retrieved knowledge?
By systematically analyzing each component, we highlight both the potential and the current shortcomings of RAG for commonsense reasoning.

The following is the repository behind the MSc thesis of Leonardo Fettucciari having Prof. Roberto Navigli as advisor and Dr. Francesco Maria Molfese as co-advisor. 
</div>

---

## 0. Installation

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install -e .
```

## 1. Assessing the Impact of Commonsense Knowledge Base Design Choices
### 1.1 If you have a KB file:
Place your `kb.jsonl` file in `data/ckb/cleaned` and skip to section 1.3.

### 1.2 Generate a KB from scratch:
To generate our Knowledge Base we use proprietary LLMs to produce commonsense statements for each noun synset/concept present in WordNet.

Inserting `gemini-1.5-flash` and `gemini-2.0-flash` in `gemini_config.json`, we run for each:
```bash
bash scripts/gemini_ckb_creation.sh <API_KEY>
```
We merge all raw corpora using:
```bash
bash scripts/ckb_merge.sh --source1 <file1.jsonl> --source2 <file2.jsonl> 
```
Apply cleanup and deduplication on merged file using:
```bash
bash scripts/ckb_cleanup.sh <input_path>
```
### 1.3 To collect RACo's KB
1. Download all files from [this link](https://drive.google.com/drive/folders/1oj2POBBy8kyBFNU5nHb05wu2DlcOfGnV), **except the CBD component**
2. Run `merge-corpus.py`
3. Rename file to `raco.jsonl`
4. Place file in `data/ckb/cleaned`

### 1.4 Evaluation
We first collect our baselines, using:
```bash
bash scripts/inference.sh --prompt-types zs,zscot --run-name baselines

bash scripts/compute_accuracy.sh --run-name baselines
```
To generate results with RACo's Knowledge Base, run:
```bash
bash scripts/inference.sh --prompt-types zscotk --retrieval-strategy-list retriever --ckb-path data/ckb/cleaned/raco.jsonl --run-name raco

bash scripts/compute_accuracy.sh --run-name raco
```
To generate results with our Knowledge Base, run:
```bash
bash scripts/inference.sh --prompt-types zscotk --retrieval-strategy-list retriever --ckb-path data/ckb/cleaned/kb.jsonl --run-name kb

bash scripts/compute_accuracy.sh --run-name kb
```
---
## 2. Investigating Retrieval Limitations for Commonsense
### 2.1 Retrieval Strategies
We start by comparing the standard and CNER retrieval strategies.
We have already collected results for the standard retrieval strategy at the previous section.

To collect results using the CNER retrieval strategy, run:
```bash
bash scripts/inference.sh --prompt-types zscotk --retrieval-strategy-list cner+retriever --ckb-path data/ckb/cleaned/kb.jsonl --run-name kb_cner

bash scripts/compute_accuracy.sh --run-name kb_cner
```

### 2.2 Deduplication
We explore deduplicating semantically similar statements in the top-k pool retrieved.

To do so we add the --rerank-type flag and collect results as:
```bash
bash scripts/inference.sh --prompt-types zscotk --retrieval-strategy-list cner+retriever --ckb-path data/ckb/cleaned/kb.jsonl --rerank-type filter --run-name kb_cner_filter

bash scripts/compute_accuracy.sh --run-name kb_cner_filter
```

### 2.3 Retriever Fine-Tuning
We iteratively fine-tune our base retriever model through the following script, setting the iteration number to 1, 2 and finally 3:
```bash
bash run_retriever_pipeline.sh --iteration <iteration_number>
```
We then use the following script to evaluate each trained retriever model:

```bash
bash scripts/inference.sh --prompt-types zscotk --retriever-model models/retriever_trained_iteration_<iteration_number> --retrieval-strategy-list cner+retriever --ckb-path data/ckb/cleaned/kb.jsonl --rerank-type filter --run-name retriever<iteration_number>

bash scripts/compute_accuracy.sh --run-name retriever<iteration_number>
```
---
## 3. How effectively do LLMs leverage retrieved knowledge?
### 3.1 Manual Annotation
We first generate the few-shot outputs with enhanced guidelines for higher explainability and an easier annotation process:
```bash
bash scripts/inference.sh --prompt-types fscot --run-name fscot
bash scripts/inference.sh --prompt-types fscotk --run-name fscotk5 --retriever-model models/retriever_trained_iteration_2 --retrieval-strategy-list cner+retriever --ckb-path data/ckb/cleaned/kb.jsonl --rerank-type filter 

bash scripts/compute_accuracy.sh --run-name fscot
bash scripts/compute_accuracy.sh --run-name fscotk5
```
Then we randomly sample entries to build our annotation corpus:
```bash
bash scripts/compare_for_annotation.sh --experiment1 fscot --prompt fscot --experiment2 fscotk5 --prompt fscotk5
```

### 3.2 Oracle-quality Knowledge
We first generate near perfect knowledge using ChatGPT's API, along a valid API key.
Set the following in `batch_api_config.yaml`:
```bash
obqa:            
  path: allenai/openbookqa
  subset: main
  split: test

qasc:            
  path: allenai/qasc
  subset: 
  split: validation

csqa:            
  path: tau/commonsense_qa
  subset:
  split: validation
```
Then to generate the knowledge, create batches, run and wait for completion:
```bash
python src/ckb_creation/create_batches.py --config_path settings/batch_api_config.yaml --dataset_path csqa --output_dir outputs/batches/oracle
python src/ckb_creation/create_batches.py --config_path settings/batch_api_config.yaml --dataset_path obqa --output_dir outputs/batches/oracle
python src/ckb_creation/create_batches.py --config_path settings/batch_api_config.yaml --dataset_path qasc --output_dir outputs/batches/oracle
```
```bash
python src/ckb_creation/batch_polling.py --input_dir outputs/batches/oracle/ --output_dir outputs/batches/oracle/results --api_key <OPENAI_API_KEY>
```
Then we use the generated knowledge as retrieved oracle-quality information replacing an external Knowledge Base.

**Note**: If you already have the oracle datasets, place them in `outputs/batches/oracle`.
```bash
bash scripts/inference_oracle.sh --run-name oracle

bash scripts/compute_accuracy.sh --run-name oracle
```
### 3.3 Leveraging Contextually Generated Knowledge
Similarly as before, we use gpt-4o-mini via API to generate knowledge statements, this time we use the training sets so not to generate oracle knowledge.
Set the following in `batch_api_config.yaml`:
```bash
obqa:            
  path: allenai/openbookqa
  subset: main
  split: train

qasc:            
  path: allenai/qasc
  subset: 
  split: train

csqa:            
  path: tau/commonsense_qa
  subset:
  split: train
```
Then to generate the knowledge, create batches, run and wait for completion:
```bash
python src/ckb_creation/create_batches.py --config_path settings/batch_api_config.yaml --dataset_path csqa --output_dir outputs/batches/contextual_ckb
python src/ckb_creation/create_batches.py --config_path settings/batch_api_config.yaml --dataset_path obqa --output_dir outputs/batches/contextual_ckb
python src/ckb_creation/create_batches.py --config_path settings/batch_api_config.yaml --dataset_path qasc --output_dir outputs/batches/contextual_ckb
```
```bash
python src/ckb_creation/batch_polling.py --input_dir outputs/batches/contextual_ckb/ --output_dir outputs/batches/contextual_ckb/results --api_key <OPENAI_API_KEY>
```
If you have the `contextual_kb.jsonl`, merge it with the `kb.jsonl` using:
```bash
bash scripts/ckb_merge.sh --source1 data/ckb/cleaned/kb.jsonl --source2 data/ckb/cleaned/contextual_kb.jsonl
```
Finally, to use the generated knowledge and evaluate the results, run:
```bash
bash scripts/inference.sh --run-name contextual --retriever-model models/retriever_trained_iteration_2 --retrieval-strategy-list cner+retriever --ckb-path data/ckb/cleaned/kb.jsonl --rerank-type filter

bash scripts/compute_accuracy.sh --run-name contextual
```
