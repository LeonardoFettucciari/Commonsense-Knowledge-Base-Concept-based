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
from src.datasets.dataset_loader import load_local_dataset, load_hf_dataset, preprocess_dataset
from src.datasets.dataset_loader import split_choices

# 0. Specify output directory
output_dir = "models/retriever_trained_all_datasets"

# 1. Load a model to finetune with model card metadata
model = SentenceTransformer(
    "intfloat/e5-base-v2",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="intfloat/e5-base-v2-trained",
    )
)

# Load datasets
csqa_llama8b_positives = load_local_dataset("data/csqa/Llama-3.1-8B-Instruct/retriever_trainset/2025-05-21_02-40-35/prompt=zscot|ckb=merged_filtered|retrieval_strategy=retriever.jsonl")
csqa_qwen7b_positives = load_local_dataset("data/csqa/Qwen2.5-7B-Instruct/retriever_trainset/2025-05-21_03-40-02/prompt=zscot|ckb=merged_filtered|retrieval_strategy=retriever.jsonl")
obqa_llama8b_positives = load_local_dataset("data/obqa/Llama-3.1-8B-Instruct/retriever_trainset/2025-05-21_12-15-43/prompt=zscot|ckb=merged_filtered|retrieval_strategy=retriever.jsonl")
obqa_qwen7b_positives = load_local_dataset("data/obqa/Qwen2.5-7B-Instruct/retriever_trainset/2025-05-21_14-11-57/prompt=zscot|ckb=merged_filtered|retrieval_strategy=retriever.jsonl")
qasc_llama8b_positives = load_local_dataset("data/qasc/Llama-3.1-8B-Instruct/retriever_trainset/2025-05-21_06-50-53/prompt=zscot|ckb=merged_filtered|retrieval_strategy=retriever.jsonl")
qasc_qwen7b_positives = load_local_dataset("data/qasc/Qwen2.5-7B-Instruct/retriever_trainset/2025-05-21_08-51-29/prompt=zscot|ckb=merged_filtered|retrieval_strategy=retriever.jsonl")


positive_datasets = [csqa_llama8b_positives, obqa_llama8b_positives, qasc_llama8b_positives,
                     csqa_qwen7b_positives, obqa_qwen7b_positives, qasc_qwen7b_positives]

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
    run_name="e5-base-mnr",
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


