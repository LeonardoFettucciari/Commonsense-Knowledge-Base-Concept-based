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

def intersect_and_align_by_id(dataset_a, dataset_b):
    ids_a = set(dataset_a["id"])
    ids_b = set(dataset_b["id"])
    common_ids = ids_a & ids_b

    dataset_a = dataset_a.filter(lambda x: x["id"] in common_ids)
    dataset_b = dataset_b.filter(lambda x: x["id"] in common_ids)

    dataset_a = dataset_a.sort("id")
    dataset_b = dataset_b.sort("id")
    return dataset_a, dataset_b



# 1. Load a model to finetune with model card metadata
model = SentenceTransformer(
    "intfloat/e5-base-v2",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="intfloat/e5-base-v2-trained-mnr",
    )
)

# Load datasets
csqa_train = load_dataset("tau/commonsense_qa", split="train")
obqa_train = load_dataset("allenai/openbookqa", split="train")
qasc_train = load_dataset("allenai/qasc",       split="train")

obqa_train = preprocess_dataset(obqa_train, "obqa")

csqa_positives = load_dataset("sapienzanlp/zebra-kb-explanations", "csqa-train-gemini", split="train")
obqa_positives = load_dataset("sapienzanlp/zebra-kb-explanations", "obqa-train-gemini", split="train")
qasc_positives = load_dataset("sapienzanlp/zebra-kb-explanations", "qasc-train-gemini", split="train")

csqa_train, csqa_positives = intersect_and_align_by_id(csqa_train, csqa_positives)
obqa_train, obqa_positives = intersect_and_align_by_id(obqa_train, obqa_positives)
qasc_train, qasc_positives = intersect_and_align_by_id(qasc_train, qasc_positives)

trainsets = [csqa_train, obqa_train, qasc_train]
positivesets = [csqa_positives, obqa_positives, qasc_positives]

pairs = []
for trainset, positiveset in zip(trainsets, positivesets):
    for i, (item, item_positives) in enumerate(zip(trainset, positiveset)):
        if item['id'] != item_positives['id']:
            raise ValueError(f"IDs do not match at index {i}: {item['id']} != {item_positives['id']}")
        
        question = item["question"]
        choices = " ".join([f"{label}. {choice}" for label, choice in zip(item['choices']['label'], item['choices']['text'])])
        anchor = f"{question} {choices}"

        positives = item_positives.get("positives", [])

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
    output_dir="models/retriever_zebra",
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
)

# 9. Train
trainer.train()

# 10. Save the trained model
model.save_pretrained("models/retriever_zebra/final")

# 11. Optional: Push to hub
# model.push_to_hub("e5-base-mnr")


