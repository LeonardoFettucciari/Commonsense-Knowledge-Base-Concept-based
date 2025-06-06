import json
import random
from datasets import load_dataset
from settings.aliases import DATASET_NAME_TO_TAG, DATASET_TAG_TO_NAME
import os

# CONFIGURATION
DATASET_NAME = "qasc"  # or any other dataset name supported by HF
SPLIT_NAME = "train"    # or "validation", "test"
dataset_name = DATASET_TAG_TO_NAME[DATASET_NAME]
dataset_tag = DATASET_NAME_TO_TAG[DATASET_NAME]
OUTPUT_FILE = f"data/{dataset_tag}/unbroken_samples.jsonl"

# Load dataset
print(f"\nLoading dataset '{dataset_name}' split '{SPLIT_NAME}'...")
dataset = load_dataset(dataset_name, split=SPLIT_NAME)
print(f"Loaded {len(dataset)} samples.\n")

# Load already selected sample ids
selected_ids = set()
if os.path.exists(OUTPUT_FILE):
    print(f"Loading already selected samples from '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            saved_sample = json.loads(line)
            selected_ids.add(saved_sample["id"])  # assumes 'id' field is unique
    print(f"Found {len(selected_ids)} previously selected samples.\n")

# Start interactive loop
print("Instructions:")
print(" - Press ENTER to skip a sample.")
print(" - Type anything and press ENTER to save the sample.")
print(" - Type 'exit' and press ENTER to quit.\n")

# Create list of not yet selected samples
remaining_samples = [s for s in dataset if s["id"] not in selected_ids]
if len(remaining_samples) == 0:
    print("✅ All samples have already been selected!")
    exit(0)

while True:
    if len(remaining_samples) == 0:
        print("✅ No more unselected samples remaining.")
        break

    # Pick a random sample not yet selected
    sample = random.choice(remaining_samples)

    # Extract fields
    question = sample.get("question") or sample.get("question_stem") or "<no question>"

    # Try to extract choices
    if "choices" in sample and isinstance(sample["choices"], dict):
        choices = sample["choices"].get("text", [])
    elif "choices" in sample and isinstance(sample["choices"], list):
        choices = sample["choices"]
    else:
        choices = []

    # Try to extract answer
    answer_text = ""
    answer_idx = None

    # Case 1: 'answerKey' as letter (A, B, C, D, ...)
    if "answerKey" in sample:
        try:
            letter_to_index = {chr(65+i): i for i in range(len(choices))}
            answer_key = sample["answerKey"]
            if answer_key in letter_to_index and letter_to_index[answer_key] < len(choices):
                answer_idx = letter_to_index[answer_key]
                answer_text = choices[answer_idx]
        except Exception:
            pass

    # Case 2: 'answer_idx' as integer index
    elif "answer_idx" in sample:
        answer_idx = sample["answer_idx"]
        if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
            answer_text = choices[answer_idx]

    # Display sample
    print("==============================")
    print(f"QUESTION:\n{question}\n")
    print("CHOICES:")
    for i, choice in enumerate(choices):
        label = chr(65+i) if i < 26 else f"({i})"  # Support >26 choices
        print(f"  {label}. {choice}")
    print(f"\nGROUND-TRUTH ANSWER: {answer_text}")
    print("==============================")

    # Prompt user
    user_input = input("Save this sample? (ENTER=skip / any text=save / 'exit'=quit): ")

    if user_input.strip().lower() == "exit":
        print("Exiting.")
        break

    elif user_input.strip() != "":
        # Save sample
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            json.dump(sample, f)
            f.write("\n")
        selected_ids.add(sample["id"])
        remaining_samples = [s for s in remaining_samples if s["id"] != sample["id"]]
        print(f"Sample saved to '{OUTPUT_FILE}'. {len(remaining_samples)} samples remaining.\n")

    else:
        print("Skipped.\n")
