#!/usr/bin/env python3
import argparse
import json
from collections import OrderedDict
from src.utils.io_utils import load_jsonl, save_jsonl

def merge_jsonl(input_paths, output_path):
    """
    Read all input JSONL files, merge entries by synset_name,
    concatenating their statements lists (preserving duplicates).
    """
    merged = OrderedDict()  # preserves insertion order of first-seen synsets

    for path in input_paths:
        data = load_jsonl(path)
        for entry in data:
            key = entry['synset_name']
            if key not in merged:
                # copy all metadata, make a shallow copy so we don't mutate original
                merged[key] = {
                    'synset_name': entry['synset_name'],
                    'synset_lemma': entry.get('synset_lemma'),
                    'synset_definition': entry.get('synset_definition'),
                    'statements': list(entry.get('statements', []))
                }
            else:
                # just extend the statements
                merged[key]['statements'].extend(entry.get('statements', []))

    # write out
    save_jsonl(list(merged.values()), output_path)
    print(f"Merged {len(input_paths)} files into {output_path}, "
          f"{len(merged)} synsets total.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL commonsense-KB dumps into one, "
                    "concatenating 'statements' per synset."
    )
    parser.add_argument(
        'inputs', nargs='+',
        help="Paths to input JSONL files to merge"
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help="Path to write merged JSONL"
    )
    args = parser.parse_args()
    merge_jsonl(args.inputs, args.output)
