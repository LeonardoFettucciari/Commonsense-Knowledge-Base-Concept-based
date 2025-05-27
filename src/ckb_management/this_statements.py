import json
from tqdm import tqdm

def extract_this_statements(input_path, output_path):
    this_statements = []

    with open(input_path, "r", encoding="utf-8") as infile:
        total_lines = sum(1 for line in infile if line.strip())
        infile.seek(0)

        for line in tqdm(infile, total=total_lines, desc="Scanning knowledge base"):
            if not line.strip():
                continue
            entry = json.loads(line)
            for s in entry.get("statements", []):
                if s.startswith("This"):
                    this_statements.append(s)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for stmt in this_statements:
            outfile.write(stmt + "\n")

    print(f"✅ Found {len(this_statements)} statements starting with 'This'")
    print(f"📝 Saved to: {output_path}")

if __name__ == "__main__":
    input_file = "data/ckb/cleaned/merged_filtered_this_fixed5.jsonl"
    output_file = "data/ckb/cleaned/statements_starting_with_this.txt"
    extract_this_statements(input_file, output_file)
