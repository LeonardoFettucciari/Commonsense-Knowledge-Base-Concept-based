import json
import random
import os
from datasets import load_dataset

# ─── CONFIGURATION ───

# Path to your input TSV file
#INPUT_PATH = "outputs/inference/csqa/Qwen2.5-7B-Instruct/zscot_augment_incorrect/2025-05-30_23-53-28/accuracy/prompt=zscot.tsv"
#INPUT_PATH = 'outputs/inference/obqa/Qwen2.5-7B-Instruct/zscot_augment_incorrect/2025-05-30_23-53-28/accuracy/prompt=zscot.tsv'
INPUT_PATH = "outputs/inference/qasc/Qwen2.5-7B-Instruct/zscot_augment_incorrect/2025-05-30_23-53-28/accuracy/prompt=zscot.tsv"
# Path to the output JSONL file (will be created if it doesn’t exist)
OUTPUT_FILE = "data/qasc/unbroken_samples_train.jsonl"

# ─── UTILITY FUNCTIONS ───

def load_saved_ids(output_path: str) -> set:
    """
    Read OUTPUT_FILE (JSONL) and return a set of all 'id' values already saved.
    """
    saved = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        saved.add(rec["id"])
                except json.JSONDecodeError:
                    continue
    return saved

def parse_choices(choices_str: str) -> dict:
    """
    Given a multiline string like:
        "A. ignore
         B. enforce
         C. authoritarian
         D. yell at
         E. avoid"
    Return a dict mapping letter -> text, e.g. {"A": "ignore", "B": "enforce", ...}.
    """
    mapping = {}
    for raw_line in choices_str.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ". " in line:
            letter, text = line.split(". ", 1)
            letter = letter.strip()
            text = text.strip()
            if len(letter) == 1 and letter.isalpha():
                mapping[letter] = text
    return mapping

def display_sample(sample: dict):
    """
    Print QUESTION, CHOICES, and GROUND-TRUTH ANSWER from one sample dict.
    """
    question = sample.get("question", "<no question>")
    choices_str = sample.get("choices", "")
    ground_truth_letter = sample.get("ground_truth", "").strip()

    choices_map = parse_choices(choices_str)

    print("\n" + "="*60)
    print("QUESTION:")
    print(question + "\n")

    print("CHOICES:")
    for letter, text in sorted(choices_map.items()):
        print(f"  {letter}. {text}")
    print()

    gt_text = choices_map.get(ground_truth_letter, "<not found>")
    print(f"GROUND-TRUTH ANSWER: {ground_truth_letter}. {gt_text}")
    print("="*60 + "\n")

def append_to_output(sample: dict, filepath: str):
    """
    Append one sample dict (as JSON) to OUTPUT_FILE.
    """
    with open(filepath, "a", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False)
        f.write("\n")


# ─── MAIN SCRIPT ───

def main():
    # 1) Load already-saved IDs
    saved_ids = load_saved_ids(OUTPUT_FILE)
    if saved_ids:
        print(f"Loaded {len(saved_ids)} previously saved IDs from '{OUTPUT_FILE}'.\n")

    # 2) Load the TSV via Hugging Face Datasets
    print(f"Loading TSV from '{INPUT_PATH}' (using 🤗 Datasets)...")
    dataset = load_dataset(
        "csv",
        data_files={ "train": INPUT_PATH },
        delimiter="\t"
    )["train"]
    total_rows = len(dataset)
    print(f"→ Total rows in TSV: {total_rows}\n")

    # 3) Filter out saved IDs and keep only rows where
    #    xfinder_extracted_answers_mismatch == 0 AND xfinder_acc_llama == 0
    def filter_fn(example):
        try:
            mismatch = int(example.get("xfinder_extracted_answers_mismatch", "1"))
            acc_llama = int(example.get("xfinder_acc_llama", "1"))
        except ValueError:
            # If conversion fails, treat as non‐zero
            return False

        if mismatch != 0 or acc_llama != 0:
            return False
        if example.get("id") in saved_ids:
            return False
        return True

    print("Applying filters (xfinder_extracted_answers_mismatch == 0 && xfinder_acc_llama == 0 && id not in saved)...")
    filtered = dataset.filter(filter_fn)
    filtered_count = len(filtered)
    print(f"→ Rows after filtering: {filtered_count}\n")

    if filtered_count == 0:
        print("✅ No samples to review after filtering. Exiting.")
        return

    # 4) Build a list of remaining indices
    remaining_indices = list(range(filtered_count))
    print("Interactive review loop starting.")
    print("  - Press ENTER to skip a sample.")
    print("  - Type anything (except 'q') + ENTER to save the sample.")
    print("  - Type 'q' + ENTER to quit.\n")

    while remaining_indices:
        # Pick a random index from the remaining_indices list
        chosen_pos = random.randrange(len(remaining_indices))
        dataset_idx = remaining_indices[chosen_pos]
        sample = filtered[dataset_idx]  # this is a dict

        display_sample(sample)

        user_in = input("Save this sample? (ENTER=skip / any text=save / 'q'=quit): ").strip()

        if user_in.lower() == "q":
            print("Exiting without saving this one.")
            break

        if user_in == "":
            # Skip: simply remove from remaining_indices and continue
            remaining_indices.pop(chosen_pos)
            print("Skipped.\n")
        else:
            # Save: append to OUTPUT_FILE, then remove from remaining_indices
            append_to_output(sample, OUTPUT_FILE)
            remaining_indices.pop(chosen_pos)
            print(f"Sample saved to '{OUTPUT_FILE}'.\n")

    if not remaining_indices:
        print("✅ All filtered samples have been processed. Exiting.")
    else:
        print("Interrupted by user. Exiting.")

if __name__ == "__main__":
    main()
