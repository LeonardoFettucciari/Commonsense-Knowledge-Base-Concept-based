import json

total_statements = 0

with open('data/ckb/cleaned/merged_filtered.jsonl', 'r') as f:
    for line in f:
        if line.strip():  # Skip empty lines
            data = json.loads(line)
            total_statements += len(data.get('statements', []))

print(f"Total number of statements: {total_statements}")
