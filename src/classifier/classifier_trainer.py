import os
import json
from collections import defaultdict

import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from torchmetrics.functional.classification import accuracy, precision, recall, f1_score

# Load tokenizer and model
model_name = "microsoft/deberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Input/output paths
input_path = "outputs/classifier/trainset_all_lemmas_wup_threshold.jsonl"
output_dir = "models/deberta-v3-classifier_trainset_all_lemmas_wup_threshold"
processed_dir = "cache/classifier/processed_datasets"
train_cache_path = os.path.join(processed_dir, "train")
eval_cache_path = os.path.join(processed_dir, "eval")

# Ensure output dirs exist
os.makedirs(processed_dir, exist_ok=True)

# Tokenization function
def preprocess(example):
    return tokenizer(
        example["synset"],
        example["statement"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

# Load raw data only if needed
need_raw_data = not (os.path.exists(train_cache_path) and os.path.exists(eval_cache_path))
if need_raw_data:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    synset_groups = defaultdict(list)
    for sample in raw_data:
        synset_groups[sample["synset"]].append(sample)

# Process or load train set
if os.path.exists(train_cache_path):
    print("Loading cached train dataset...")
    tok_train_dataset = Dataset.load_from_disk(train_cache_path)
else:
    print("Processing train dataset...")
    train_data = []
    for synset, samples in synset_groups.items():
        if len(samples) == 12:
            eval_indices = [0, 4, 10, 11]
            train_data.extend([s for i, s in enumerate(samples) if i not in eval_indices])
        else:
            train_data.extend(samples)

    train_dataset = Dataset.from_list(train_data)
    tok_train_dataset = train_dataset.map(preprocess, batched=True)
    tok_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tok_train_dataset.save_to_disk(train_cache_path)

# Process or load eval set
if os.path.exists(eval_cache_path):
    print("Loading cached eval dataset...")
    tok_eval_dataset = Dataset.load_from_disk(eval_cache_path)
else:
    print("Processing eval dataset...")
    eval_data = []
    for synset, samples in synset_groups.items():
        if len(samples) == 12:
            eval_indices = [0, 4, 10, 11]
            eval_data.extend([samples[i] for i in eval_indices])

    eval_dataset = Dataset.from_list(eval_data)
    tok_eval_dataset = eval_dataset.map(preprocess, batched=True)
    tok_eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tok_eval_dataset.save_to_disk(eval_cache_path)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    bf16=True,  # Enable if supported
)

# Custom metrics function
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=-1)
    labels = torch.tensor(p.label_ids)
    return {
        "accuracy": accuracy(preds, labels, task="binary").item(),
        "precision": precision(preds, labels, task="binary").item(),
        "recall": recall(preds, labels, task="binary").item(),
        "f1": f1_score(preds, labels, task="binary").item(),
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_train_dataset,
    eval_dataset=tok_eval_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()

# Print evaluation results
print(f"Accuracy:  {results['eval_accuracy']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall:    {results['eval_recall']:.4f}")
print(f"F1-score:  {results['eval_f1']:.4f}")

# Save final model
model.save_pretrained(f"{output_dir}/final")
