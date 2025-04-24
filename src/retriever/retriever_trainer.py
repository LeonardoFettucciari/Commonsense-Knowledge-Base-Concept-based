from datasets import load_dataset, concatenate_datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

from src.datasets.dataset_loader import load_local_dataset

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "intfloat/e5-base-v2",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="intfloat/e5-base-v2-trained",
    )
)

# 3. Load a dataset to finetune on
csqa = load_local_dataset("outputs/retriever/training_data/final/csqa.jsonl")
obqa = load_local_dataset("outputs/retriever/training_data/final/obqa.jsonl")
qasc = load_local_dataset("outputs/retriever/training_data/final/qasc.jsonl")
dataset = concatenate_datasets([csqa, obqa, qasc])

split = dataset.train_test_split(0.1, shuffle=True, seed=42)
train_dataset = split['train']
eval_dataset = split["test"]



triplets = []
for sample in train_dataset:
    anchor = sample.get("question") + " " + " ".join(sample.get("choices").split("\n"))
    positives = sample.get("positives", [])
    negatives = sample.get("negatives", [])

    for pos in positives:
        for neg in negatives:
            triplet = {
                "anchor": anchor,
                "positive": pos,
                "negative": neg,
            }
            triplets.append(triplet)
train_dataset = dataset.from_list(triplets)

triplets = []
for sample in eval_dataset:
    anchor = sample.get("question") + " " + " ".join(sample.get("choices").split("\n"))
    positives = sample.get("positives", [])
    negatives = sample.get("negatives", [])

    for pos in positives:
        for neg in negatives:
            triplet = {
                "anchor": anchor,
                "positive": pos,
                "negative": neg,
            }
            triplets.append(triplet)
eval_dataset = dataset.from_list(triplets)

# 4. Define a loss function
loss = TripletLoss(model)


num_train_epochs = 5
total_steps = len(train_dataset) // 256 * num_train_epochs
warmup_steps = int(0.1 * total_steps)
# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Core settings
    output_dir="models/retriever",
    num_train_epochs=num_train_epochs,                  # 3 full passes over your training data
    per_device_train_batch_size=64,      # push this up if you have >16 GB GPU
    per_device_eval_batch_size=64,       # larger eval batches to speed up validation

    # Optimization
    learning_rate=3e-5,                  # a good starting LR for transformers
    weight_decay=0.01,                   # regularization on weights
    lr_scheduler_type="linear",          # linear warmup then decay
    warmup_steps=500,                    # ~10% of a 5 k-step run

    # Mixed precision
    fp16=True,                           # if your GPU supports it, speeds up training
    bf16=False,                          # disable unless you know your GPU supports BF16

    # Logging & checkpointing
    logging_strategy="steps",
    logging_steps=100,                   # log loss every 100 steps
    save_strategy="steps",
    save_steps=500,                      # checkpoint every 500 steps
    save_total_limit=3,                  # keep only the last 3 checkpoints
    load_best_model_at_end=True,         # after training, reload best checkpoint
    metric_for_best_model="eval_loss",   # choose best checkpoint by lowest eval loss

    # Evaluation
    eval_strategy="steps",
    eval_steps=500,                      # run evaluation every 500 steps

    # Reproducibility & performance
    seed=42,
    remove_unused_columns=True,          # drop unused dataset cols

    # Experiment tracking (e.g., wandb)
    run_name="e5-base-v2-triplet-finetune",
)


# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset[:]["anchor"],
    positives=eval_dataset[:]["positive"],
    negatives=eval_dataset[:]["negative"],
    name="all-nli-dev",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

'''
# (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="all-nli-test",
)
test_evaluator(model)
'''
# 8. Save the trained model
model.save_pretrained("models/retriever/final")

# 9. (Optional) Push it to the Hugging Face Hub
#model.push_to_hub("mpnet-base-all-nli-triplet")