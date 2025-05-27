import inflect
import json
import re

# Initialize inflect engine
p = inflect.engine()

def replace_they_with_plural_lemma(entry):
    lemma = entry["synset_lemma"]
    plural = p.plural(lemma)

    updated_statements = []
    for s in entry["statements"]:
        if re.match(r'^They\b', s):
            s = re.sub(r'^They\b', plural.capitalize(), s)
        updated_statements.append(s)

    entry["statements"] = updated_statements
    return entry

def process_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if line.strip():  # skip empty lines
                entry = json.loads(line)
                updated_entry = replace_they_with_plural_lemma(entry)
                json.dump(updated_entry, outfile)
                outfile.write("\n")

# Example usage
if __name__ == "__main__":
    input_file = "data/ckb/cleaned/merged_filtered.jsonl"
    output_file = "data/ckb/cleaned/merged_filtered_fixed.jsonl"
    process_jsonl(input_file, output_file)
    print(f"Processed entries saved to: {output_file}")
