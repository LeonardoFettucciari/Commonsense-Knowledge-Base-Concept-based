#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def count_statements(input_path: Path):
    total_statements = 0
    synset_count = 0

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")
                continue

            statements = record.get("statements", [])
            if not isinstance(statements, list):
                continue

            total_statements += len(statements)
            synset_count += 1

    print(f"Processed {synset_count} synsets")
    print(f"Total statements: {total_statements}")

def main():
    parser = argparse.ArgumentParser(description="Count statements in a JSONL KB file")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSONL file")
    args = parser.parse_args()
    count_statements(Path(args.input))

if __name__ == "__main__":
    main()
