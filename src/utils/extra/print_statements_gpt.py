#!/usr/bin/env python3
import json
import argparse
import sys

def split_and_print(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                content = (
                    record
                    .get('response', {})
                    .get('body', {})
                    .get('choices', [])[0]
                    .get('message', {})
                    .get('content', '')
                )
            except (json.JSONDecodeError, IndexError, AttributeError) as e:
                print(f"Warning: could not parse line {line_no}: {e}", file=sys.stderr)
                continue

            # Split on [SEP] and print non-empty statements
            print(f"\n\nLine {line_no}:")
            for stmt in content.split('[SEP]'):
                stmt = stmt.strip()
                if stmt:
                    print(stmt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read a JSONL-like file, split assistant content on [SEP], and print each statement.'
    )
    parser.add_argument('input_file', help='Path to the input file')
    args = parser.parse_args()
    split_and_print(args.input_file)
