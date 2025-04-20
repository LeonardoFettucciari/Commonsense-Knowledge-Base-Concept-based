from datasets import load_dataset, concatenate_datasets, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

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

# 2. Load the triplet datasets
csqa = load_local_dataset("outputs/retriever/training_data/final/MNR_csqa.jsonl")
obqa = load_local_dataset("outputs/retriever/training_data/final/MNR_obqa.jsonl")
qasc = load_local_dataset("outputs/retriever/training_data/final/MNR_qasc.jsonl")
dataset = concatenate_datasets([csqa, obqa, qasc])


# 5. Train/test split
split = dataset.train_test_split(0.1, shuffle=True, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# 6. Define the MNR loss
loss = MultipleNegativesRankingLoss(model)

# 7. Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="models/retriever_mnr",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="e5-base-mnr",
    
)

# 8. Trainer (no TripletEvaluator; can add IR evaluator here)
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
model.save_pretrained("models/retriever_mnr/final")

# 11. Optional: Push to hub
# model.push_to_hub("e5-base-mnr")
