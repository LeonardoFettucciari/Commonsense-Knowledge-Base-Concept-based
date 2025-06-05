import logging
import traceback

from datasets import load_dataset

from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import MultipleNegativesRankingLoss

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
train_batch_size = 128
num_epochs = 1
num_rand_negatives = 5  # How many random negatives should be used for each question-answer pair
full_dataset_path="outputs/retriever/crossencoder/positive_pairs|name|lemma|definition.jsonl"

# 1a. Load a model to finetune with 1b. (Optional) model card data
model = CrossEncoder(
    model_name,
    model_card_data=CrossEncoderModelCardData(
        language="en",
        license="apache-2.0",
        model_name="ms-marco-MiniLM-L6-v2 trained on (statement, synset+lemma+definition, label) triples from ckb",
    ),
)
print("Model max length:", model.max_length)
print("Model num labels:", model.num_labels)

# 2. Load the dataset
logging.info("Read the gooaq training dataset")
full_dataset = load_dataset(
    "json",
    data_files=full_dataset_path,
    split="train"          
)
splits = full_dataset.train_test_split(0.1, shuffle=True, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]
logging.info(train_dataset)
logging.info(eval_dataset)

# 3. Define our training loss.
loss = MultipleNegativesRankingLoss(
    model=model,
    num_negatives=num_rand_negatives,
)

# 4. Use CrossEncoderNanoBEIREvaluator, a light-weight evaluator for English reranking
#evaluator = CrossEncoderNanoBEIREvaluator(
#    dataset_names=["msmarco", "nfcorpus", "nq"],
#    batch_size=train_batch_size,
#)
#evaluator(model)

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"classifier-{short_model_name}-name|lemma|definition"
args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=1e-6,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=4000,
    save_strategy="steps",
    save_steps=4000,
    save_total_limit=2,
    logging_steps=500,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    seed=42,
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    #evaluator=evaluator,
)
trainer.train()

# 7. Evaluate the final model, useful to include these in the model card
#evaluator(model)

# 8. Save the final model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)