from datasets import load_dataset, Dataset

# Load the original JSONL
dataset = load_dataset(
    "json",
    data_files="outputs/retriever/training_data/qasc/Llama-3.1-8B-Instruct/training_data/prompt=trainset_retriever|ckb=full_ckb|retrieval_strategy=retriever.jsonl",
    split="train"
)

triplet_dataset = []

for sample in dataset:
    for negative in sample['negative']:
        triplet_dataset.append({
            "anchor": sample["anchor"],
            "positive": sample["positive"],
            "negative": negative,
        })

triplet_dataset = dataset.from_list(triplet_dataset)

# Save it if needed
triplet_dataset.to_json("outputs/retriever/training_data/qasc/Llama-3.1-8B-Instruct/training_data/triplets.jsonl")
