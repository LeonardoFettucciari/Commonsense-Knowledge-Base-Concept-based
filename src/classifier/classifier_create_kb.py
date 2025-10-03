import torch
import os
import json
from tqdm import tqdm
from src.utils.model_utils import get_ner_pipeline
from src.utils.data_utils import synsets_from_batch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from nltk.corpus import wordnet as wn
from src.classifier.create_trainingdata_classifier import build_ctx_noun
from src.utils.io_utils import load_local_file
from collections import defaultdict
import glob

# Settings
model_path = "models/deberta-v3-classifier_wup/final"
BATCH_SIZE = 8
NER_BATCH_SIZE = 64
MAX_TEST_SAMPLES_PER_FILE = None  # Set to None for full processing

# Load trained model
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ner_pipeline = get_ner_pipeline("Babelscape/cner-base")

# Prepare output dir
os.makedirs("outputs/batches/contextual_ckb/results", exist_ok=True)

# Find input jsonl files
jsonl_files = glob.glob("outputs/batches/contextual_ckb/*.jsonl")
print(f"Found {len(jsonl_files)} input JSONL files.")

# Process each jsonl file individually
for input_path in tqdm(jsonl_files, desc="Processing JSONL files"):
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"outputs/batches/contextual_ckb/results/{input_filename}.jsonl"

    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}, skipping...")
        continue

    print(f"\nProcessing file: {input_path}")
    print(f"Output will be saved to: {output_path}")

    # Load ckb_statements from this file
    data = load_local_file(input_path)
    sentences = []
    for item in tqdm(data, desc=f"Loading sentences from {input_filename}"):
        sentences.extend(item["ckb_statements"])

    if len(sentences) == 0:
        print(f"No sentences found in {input_path}, skipping...")
        continue

    # Limit samples for testing
    if MAX_TEST_SAMPLES_PER_FILE is not None:
        sentences = sentences[:MAX_TEST_SAMPLES_PER_FILE]
        print(f"Limiting to {len(sentences)} samples for testing.")

    # Synset extraction
    synsets_per_sample = list(
        synsets_from_batch(sentences, ner_pipeline, batch_size=NER_BATCH_SIZE)
    )

    # Collect statements for best synset
    synset_to_statements = defaultdict(list)
    for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc="Scoring batches"):
        batch_sentences = sentences[i:i + BATCH_SIZE]
        batch_synsets = synsets_per_sample[i:i + BATCH_SIZE]

        batch_sentence_batch = []
        batch_synset_ctx = []
        batch_synsets_refs = []

        for sentence, synsets in zip(batch_sentences, batch_synsets):
            if not synsets:
                continue

            synset_ctx = [build_ctx_noun(syn.name()) for syn in synsets]
            sentence_batch = [sentence] * len(synsets)

            batch_sentence_batch.extend(sentence_batch)
            batch_synset_ctx.extend(synset_ctx)
            batch_synsets_refs.append((sentence, synsets))

        if not batch_sentence_batch:
            continue

        # Run model on batch
        encodings = tokenizer(
            batch_synset_ctx,
            batch_sentence_batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**encodings).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

        # Update synset_to_statements
        idx = 0
        for sentence, synsets in batch_synsets_refs:
            n_syns = len(synsets)
            syn_probs = probs[idx:idx + n_syns]

            best_synset = synsets[syn_probs.index(max(syn_probs))]
            synset_to_statements[best_synset.name()].append(sentence)

            idx += n_syns

        # Cleanup
        del encodings
        del logits
        torch.cuda.empty_cache()

    # Write result for this file
    with open(output_path, "w", encoding="utf-8") as f:
        for synset_name in tqdm(sorted(synset_to_statements.keys()), desc="Writing JSONL"):
            syn = wn.synset(synset_name)
            row = {
                "synset_name": syn.name(),
                "synset_lemma": syn.lemmas()[0].name(),
                "synset_definition": syn.definition(),
                "statements": synset_to_statements[synset_name]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Output saved to: {output_path}")

    # Final full GPU cache cleanup
    torch.cuda.empty_cache()

print("All files processed.")
