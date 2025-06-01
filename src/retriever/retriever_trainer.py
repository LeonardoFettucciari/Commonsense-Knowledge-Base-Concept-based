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
import glob
import os
from src.datasets.dataset_loader import load_local_dataset, load_hf_dataset, preprocess_dataset
from src.datasets.dataset_loader import split_choices



# 0. Specify input and output paths
RUN_NAME = "iteration_2"
BASE_DIR = "outputs/retriever_trainset"
GLOB_PATTERN = f"{BASE_DIR}/*/*/{RUN_NAME}_positives_negatives/*/*.jsonl"
output_dir = f"models/retriever_trained_{RUN_NAME}"

# 1. Load a model to finetune with model card metadata
model = SentenceTransformer(
    "intfloat/e5-base-v2",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="intfloat/e5-base-v2-trained",
    )
)

# Load training datasets

# Filter: include only files that start with 'prompt=' and exclude 'triplets_'
all_files = sorted([
    path for path in glob.glob(GLOB_PATTERN)
    if os.path.basename(path).startswith("prompt=")
])

print(f"Found {len(all_files)} valid input files:")
for f in all_files:
    print("→", f)

# Load datasets
positive_datasets = [load_local_dataset(path) for path in all_files]



pairs = []
for positive_dataset in positive_datasets:
    for item in positive_dataset:
        question = item["question"]
        choices = item["choices"]
        anchor = f"{question}\n{choices}"
        positives = item.get("positives", [])

        for pos in positives:
            row = {
                "anchor": f"query: {anchor}",
                "positive": f"passage: {pos}"
           }
            pairs.append(row)
train_dataset = Dataset.from_list(pairs)

# 5. Train/test split
split = train_dataset.train_test_split(0.1, shuffle=True, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


# 6. Define the MNR loss
loss = MultipleNegativesRankingLoss(model)

# 7. Training arguments
num_train_epochs = 5
batch_size = 256
total_steps = len(train_dataset) // batch_size * num_train_epochs
warmup_steps = int(0.1 * total_steps)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,    # try fewer and watch eval loss
    per_device_train_batch_size=batch_size,      # halves memory, still plenty of negatives
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,                   # a bit lower for contrastive
    warmup_steps=warmup_steps,  # 10% of all steps
    weight_decay=0.01,
    fp16=True,                # enable fp16 if supported
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="epoch",
    eval_steps=None,           # evaluate less often
    save_strategy="epoch",
    save_steps=None,
    save_total_limit=2,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    run_name=RUN_NAME,
)

# 8. Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)

# 9. Train
trainer.train()

# 10. Save the trained model
model.save_pretrained(f"{output_dir}/final")

# 11. Optional: Push to hub
# model.push_to_hub("e5-base-mnr")


