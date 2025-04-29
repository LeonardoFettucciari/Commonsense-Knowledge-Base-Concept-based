from datasets import load_dataset, concatenate_datasets, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import RerankingEvaluator

from src.datasets.dataset_loader import load_local_dataset


# 1. Load a model to finetune with model card metadata
model = SentenceTransformer(
    "intfloat/e5-base-v2",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="intfloat/e5-base-v2-trained-mnr",
    )
)

# 2. Load the datasets
csqa = load_local_dataset("outputs/retriever/training_data/final/csqa.jsonl")
obqa = load_local_dataset("outputs/retriever/training_data/final/obqa.jsonl")
qasc = load_local_dataset("outputs/retriever/training_data/final/qasc.jsonl")
dataset = concatenate_datasets([csqa, obqa, qasc])

# 5. Train/test split
split = dataset.train_test_split(0.1, shuffle=True, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


# 2.1 Process datasets
processed_samples = []
for item in train_dataset:
    question = item["question"]
    choices = " ".join(item["choices"].split("\n"))
    anchor = f"{question} {choices}"

    positives = item.get("positives", [])
    negatives = item.get("negatives", [])

    if not positives:
        continue

    for pos in positives:
        for neg in negatives:
            row = {
                "anchor": anchor,
                "positive": pos,
                "negative": neg,
            }
            processed_samples.append(row)
# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(processed_samples)

processed_samples = []
for item in eval_dataset:
    question = item["question"]
    choices = " ".join(item["choices"].split("\n"))
    query = f"{question} {choices}"

    positives = item.get("positives", [])
    negatives = item.get("negatives", [])

    if not positives:
        continue

    for pos in positives:
        for neg in negatives:
            row = {
                "query": query,
                "positive": pos,
                "negative": neg,
            }
            processed_samples.append(row)
# Convert to HuggingFace Dataset
eval_dataset = Dataset.from_list(processed_samples)

reranking_evaluator = RerankingEvaluator(
    samples=eval_dataset,
    name="eval_dataset_evaluator",
    show_progress_bar=True
)
results = reranking_evaluator(model)

# 6. Define the MNR loss
loss = MultipleNegativesRankingLoss(model)

# 7. Training arguments
num_train_epochs = 5
batch_size = 256
total_steps = len(train_dataset) // batch_size * num_train_epochs
warmup_steps = int(0.1 * total_steps)
args = SentenceTransformerTrainingArguments(
    output_dir="models/retriever_mnr",
    num_train_epochs=num_train_epochs,    # try fewer and watch eval loss
    per_device_train_batch_size=batch_size,      # halves memory, still plenty of negatives
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,                   # a bit lower for contrastive
    warmup_steps=warmup_steps,  # 10% of all steps
    weight_decay=0.01,
    fp16=True,                # enable fp16 if supported
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=500,           # evaluate less often
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    run_name="e5-base-mnr",
)

# 8. Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=reranking_evaluator,
)

# 9. Train
trainer.train()

# 10. Save the trained model
model.save_pretrained("models/retriever_mnr/final")

# 11. Optional: Push to hub
# model.push_to_hub("e5-base-mnr")
