import logging
import torch
from collections import Counter

from datasets import load_dataset

from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

# Set the log level to INFO to get more information
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
train_batch_size = 128
num_epochs = 3
full_dataset_path = "outputs/classifier/full_gloss.jsonl"
short_model_name = model_name.split("/")[-1]
run_name = f"classifier-{short_model_name}-pos_weighted"
max_train_samples = 1000000  

# 1a. Load a model to finetune with 1b. (Optional) model card data
model = CrossEncoder(
    model_name,
    max_length=256,
    model_card_data=CrossEncoderModelCardData(
        language="en",
        license="apache-2.0",
        model_name=(
            "ms-marco-MiniLM-L6-v2 trained on "
            "(statement, synset_ctx, label) triples from ckb"
        ),
    ),
)
print("Model max length:", model.max_length)
print("Model num labels:", model.num_labels)

# 2. Load the dataset
logging.info("Read the training dataset")
from datasets import load_dataset, Dataset
import random

# ------------------------------------------------------------------
# 0.  Load your file as a Dataset
# ------------------------------------------------------------------
ds = load_dataset(
        "json", 
        data_files=full_dataset_path,
        split="train"                     
)

# ------------------------------------------------------------------
# 1.  Select evaluation sentences
# ------------------------------------------------------------------
import random
from collections import defaultdict

RNG = random.Random(42)          # reproducible

# 1a.  build:  synset_id  →  {sentences that are positive for it}
pos_sents_by_syn = defaultdict(set)
neg_sents_by_syn = defaultdict(set)


for synset_str, sent, lab in zip(ds["synset"], ds["statement"], ds["label"]):
    if lab == 1.0:
        syn_id = synset_str.split()[0]          # e.g. "disaster.n.03"
        pos_sents_by_syn[syn_id].add(sent)
    else:
        syn_id = synset_str.split()[0]          # e.g. "disaster.n.03"
        neg_sents_by_syn[syn_id].add(sent)

# 1b. choose ONE positive sentence per synset
eval_statements = set()
for syn_id, sentences in pos_sents_by_syn.items():
    while(True):
        chosen_sent = RNG.choice(list(sentences))
        if chosen_sent not in eval_statements:
            eval_statements.add(chosen_sent)
            break

logging.info(f"Selected {len(eval_statements):,} eval sentences "
             f"(one for each of {len(pos_sents_by_syn):,} synsets)")

# ------------------------------------------------------------------
# 2.  Split the Dataset
# ------------------------------------------------------------------
eval_dataset  = ds.filter(
    lambda ex: ex["statement"] in eval_statements,
    num_proc=4,
    keep_in_memory=True,
)

train_dataset = ds.filter(
    lambda ex: ex["statement"] not in eval_statements,
    num_proc=4,
    keep_in_memory=True,
)

# ------------------------------------------------------------------
# 3.  Shuffle and print label statistics
# ------------------------------------------------------------------
train_dataset = train_dataset.shuffle(seed=42)
eval_dataset  = eval_dataset.shuffle(seed=42)

from collections import Counter
train_dist = Counter(train_dataset["label"])
eval_dist  = Counter(eval_dataset["label"])

logging.info(f"Train label distribution: {train_dist}")
logging.info(f"Eval  label distribution:  {eval_dist}")


# Calculate weight: more weight to positive samples
num_pos = train_dist[1.0]
num_neg = train_dist[0.0]
pos_weight_value = num_neg / num_pos
# Must be a 1D tensor
pos_weight_tensor = torch.tensor([pos_weight_value])

# 3. Define our training loss.
loss = BinaryCrossEntropyLoss(model=model, pos_weight=pos_weight_tensor)


# 4. Use CrossEncoderNanoBEIREvaluator, a light-weight evaluator for English reranking
# evaluator = CrossEncoderNanoBEIREvaluator(
#     dataset_names=["msmarco", "nfcorpus", "nq"],
#     batch_size=train_batch_size,
# )
# evaluator(model)

# 5. Define the training arguments

args = CrossEncoderTrainingArguments(
    output_dir=f"models/{run_name}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=500,
    logging_first_step=True,
    run_name=run_name,
    seed=42,
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    # evaluator=evaluator,
)
trainer.train()

# 7. Evaluate the final model, useful to include these in the model card
# evaluator(model)

# 8. Save the final model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)
