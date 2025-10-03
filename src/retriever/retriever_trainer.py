from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
import glob
import os
import logging
from argparse import ArgumentParser
from src.datasets.dataset_loader import load_local_dataset


def train_retriever(
    retriever_model_to_finetune: str,
    run_name: str,
    trainset_base_dir: str,
    output_dir: str,
) -> None:
    
    # Specify training data input paths
    GLOB_PATTERN = f"{trainset_base_dir}/*/*/{run_name}_positives_negatives/*/*.jsonl"

    # Load a model to finetune with model card metadata
    model = SentenceTransformer(
        retriever_model_to_finetune,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{retriever_model_to_finetune}_{run_name}",
        )
    )

    # Load training datasets and apply preprocessing
    all_files = sorted([
        path for path in glob.glob(GLOB_PATTERN)
        if os.path.basename(path).startswith("prompt=")
    ])
    print(f"Found {len(all_files)} valid input files:")
    for f in all_files:
        print(f)
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

    # Train/test split
    split = train_dataset.train_test_split(0.1, shuffle=True, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Define the MNR loss
    loss = MultipleNegativesRankingLoss(model)

    # Training arguments
    num_train_epochs = 5
    batch_size = 256
    total_steps = len(train_dataset) // batch_size * num_train_epochs
    warmup_steps = int(0.1 * total_steps)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,    
        per_device_train_batch_size=batch_size,      
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,                   
        warmup_steps=warmup_steps,  
        weight_decay=0.01,
        fp16=True,                
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        eval_steps=None,           
        save_strategy="epoch",
        save_steps=None,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        run_name=run_name,
    )

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    # Train
    trainer.train()

    # Save the trained model
    model.save_pretrained(f"{output_dir}/final")


def main() -> None:
    parser = ArgumentParser(description="Retriever training.")
    parser.add_argument("--retriever_model_to_finetune", type=str, required=True,
                        help="Base model to fine-tune.")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Iteration tag.")
    parser.add_argument("--trainset_base_dir", type=str, required=True,
                        help="Training data positives/negatives root.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root path to store outputs.")
    args = parser.parse_args()

    logging.info("Launching retriever trainer with args: %s", vars(args))
    train_retriever(
        retriever_model_to_finetune=args.retriever_model_to_finetune,
        run_name=args.run_name,
        trainset_base_dir=args.trainset_base_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()