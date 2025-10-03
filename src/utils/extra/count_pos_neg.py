#!/usr/bin/env python3
import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Count positives and negatives in each JSONL row and overall totals."
    )
    parser.add_argument(
        "jsonl_file",
        help="Path to the input JSONL file"
    )
    args = parser.parse_args()

    total_positives = 0
    total_negatives = 0

    with open(args.jsonl_file, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {lineno}: JSON decode error ({e})")
                continue

            positives = item.get("positives", [])
            negatives = item.get("negatives", [])

            n_pos = len(positives)
            n_neg = len(negatives)

            total_positives += n_pos
            total_negatives += n_neg

            print(f"{item.get('id', f'line_{lineno}')} ,  positives: {n_pos}, negatives: {n_neg}")

    print("\n=== Totals ===")
    print(f"Total positives: {total_positives}")
    print(f"Total negatives: {total_negatives}")

if __name__ == "__main__":
    main()
