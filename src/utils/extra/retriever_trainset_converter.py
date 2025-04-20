import json
from pathlib import Path
from datasets import Dataset

def convert_triplet_jsonl_to_mnr(input_path: str):
    """
    Converts a triplet-formatted JSONL file into a JSONL compatible with MultipleNegativesRankingLoss.
    Writes the result as MNR_<original_filename>.jsonl in the same directory.

    :param input_path: Path to a JSONL file with fields: 'anchor', 'positive', 'negative'
    """
    input_file = Path(input_path)
    assert input_file.exists(), f"Input file {input_file} does not exist."

    # Load and deduplicate
    seen = set()
    anchor_positive_pairs = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pair = (obj["anchor"], obj["positive"])
            if pair not in seen:
                seen.add(pair)
                anchor_positive_pairs.append({"anchor": pair[0], "positive": pair[1]})

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(anchor_positive_pairs)

    # Write to new file
    output_file = input_file.parent / f"MNR_{input_file.name}"
    dataset.to_json(str(output_file), orient="records", lines=True, force_ascii=False)
    print(f"✔ Saved MNR-formatted file to: {output_file}")

# Example usage:
convert_triplet_jsonl_to_mnr("outputs/retriever/training_data/final/csqa.jsonl")
convert_triplet_jsonl_to_mnr("outputs/retriever/training_data/final/obqa.jsonl")
convert_triplet_jsonl_to_mnr("outputs/retriever/training_data/final/qasc.jsonl")
