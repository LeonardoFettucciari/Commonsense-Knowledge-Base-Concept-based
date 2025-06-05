from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from torchmetrics.functional.classification import accuracy, precision, recall, f1_score
import json
from collections import defaultdict

# === 1. Reload the model and tokenizer ===
model_path = "models/deberta-v3-classifier_trainset_all_lemmas_wup_threshold/final"
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)

# === 2. Re-load the eval dataset ===
with open("outputs/classifier/trainset_all_lemmas_wup_threshold.jsonl", "r") as f:
    raw_data = [json.loads(line) for line in f]

synset_groups = defaultdict(list)
for sample in raw_data:
    synset_groups[sample["synset"]].append(sample)

eval_data = []
for synset, samples in synset_groups.items():
    if len(samples) == 12:
        eval_indices = [0, 4, 10, 11]
        eval_data.extend([samples[i] for i in eval_indices])

eval_dataset = Dataset.from_list(eval_data)

def preprocess(example):
    return tokenizer(
        example["synset"],
        example["statement"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tok_eval_dataset = eval_dataset.map(preprocess, batched=True)
tok_eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# === 3. Define training args and metrics ===
training_args = TrainingArguments(
    output_dir="./eval_tmp",  # Doesn't need to be the training dir
    per_device_eval_batch_size=32,
    bf16=True,
)

def compute_metrics(p):
    import torch
    preds = torch.argmax(torch.tensor(p.predictions), axis=-1)
    labels = torch.tensor(p.label_ids)
    return {
        "accuracy": accuracy(preds, labels, task="binary").item(),
        "precision": precision(preds, labels, task="binary").item(),
        "recall": recall(preds, labels, task="binary").item(),
        "f1": f1_score(preds, labels, task="binary").item(),
    }

# === 4. Create Trainer and evaluate ===
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tok_eval_dataset,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall: {results['eval_recall']:.4f}")
print(f"F1-score: {results['eval_f1']:.4f}")
