#!/usr/bin/env python3
"""
Filter duplicate statements (exact + semantic) inside each synset of a commonsense KB,
producing in a default directory (`data/ckb/cleaned` unless overridden):
  1. `<input_basename>_filtered.jsonl` — the filtered KB JSONL.
  2. `filtering_summary.jsonl` — a single-line JSONL with total removal counts.
  3. `filtering_details.jsonl` — JSONL details for synsets with removals.
Includes logging and progress bars to track processing.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter KB statements (exact & semantic duplicates)"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input JSONL KB file"
    )
    parser.add_argument(
        "-O", "--out-dir", default="data/ckb/cleaned",
        help="Directory for all outputs (will be created)"
    )
    parser.add_argument(
        "-m", "--model", default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model name or path"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.85,
        help="Cosine similarity threshold for semantic dedupe"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=64,
        help="Batch size for embedding computation"
    )
    return parser.parse_args()


def load_model(name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(name, device=device)
    logging.info(f"Loaded SBERT model '{name}' on {device}")
    return model


def dedupe_statements(
    statements: List[str],
    model: SentenceTransformer,
    threshold: float,
    batch_size: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Remove exact and semantic duplicates from a list of statements.

    Args:
      statements: List of input text strings.
      model: Pretrained SentenceTransformer model used for encoding text into embeddings.
      threshold: Cosine similarity threshold to consider two statements as semantically similar.
      batch_size: Batch size for encoding statements.

    Returns:
      Tuple containing:
        - kept: List of statements kept after deduplication.
        - removed_exact: List of statements removed because they were exact duplicates.
        - removed_semantic: List of statements removed because they were semantically similar to an already kept statement.
    """

    # Preprocess statements: remove empty strings and strip whitespace
    cleaned = [s.strip() for s in statements if s and s.strip()]

    # Exact deduplication
    seen = set()           # Tracks statements already encountered
    unique = []            # List of unique (exact) statements
    removed_exact = []     # List of statements that were exact duplicates

    for s in cleaned:
        if s in seen:
            removed_exact.append(s)   # If already seen, mark as duplicate
        else:
            seen.add(s)
            unique.append(s)          # Otherwise, keep it

    # If 0 or 1 unique statements, no semantic deduplication needed
    removed_semantic = []
    if len(unique) <= 1:
        return unique, removed_exact, removed_semantic

    # Encode unique statements into embeddings
    embeds = model.encode(
        unique,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=False
    )

    # Semantic deduplication
    kept, kept_embs = [], []  # Lists for statements and their embeddings to keep

    for text, emb in zip(unique, embeds):
        if not kept_embs:
            # Always keep the first statement
            kept.append(text)
            kept_embs.append(emb)
        else:
            # Compare current embedding to all previously kept embeddings
            if util.cos_sim(emb, torch.stack(kept_embs)).max().item() < threshold:
                kept.append(text)
                kept_embs.append(emb)
            else:
                # If similar enough (cosine similarity >= threshold), remove it
                removed_semantic.append(text)

    return kept, removed_exact, removed_semantic



def filter_kb(
    input_path: Path,
    out_dir: Path,
    model: SentenceTransformer,
    threshold: float,
    batch_size: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = input_path.stem
    kb_out = out_dir / f"{base}_filtered.jsonl"
    summary_out = out_dir / "filtering_summary.jsonl"
    details_out = out_dir / "filtering_details.jsonl"

    total_exact = 0
    total_semantic = 0
    synset_count = 0

    # Count total lines for progress bar (optional, warning: extra pass)
    try:
        total_lines = sum(1 for _ in input_path.open('r', encoding='utf-8'))
    except Exception:
        total_lines = None

    with input_path.open('r', encoding='utf-8') as fin, \
         kb_out.open('w', encoding='utf-8') as fout_kb, \
         details_out.open('w', encoding='utf-8') as fout_det:
        iterator = tqdm(fin, total=total_lines, desc="Filtering synsets", unit="synset")
        for line in iterator:
            if not line.strip():
                continue
            synset_count += 1
            record = json.loads(line)
            kept, rem_ex, rem_sm = dedupe_statements(
                record.get('statements', []), model,
                threshold, batch_size
            )
            # write filtered KB
            record['statements'] = kept
            fout_kb.write(json.dumps(record, ensure_ascii=False) + '\n')

            # accumulate and write details
            if rem_ex or rem_sm:
                total_exact += len(rem_ex)
                total_semantic += len(rem_sm)
                det = {
                    'synset_name': record.get('synset_name'),
                    'kept_statements': kept,
                    'removed_statements': rem_ex + rem_sm
                }
                fout_det.write(json.dumps(det, ensure_ascii=False) + '\n')

    # write summary JSONL
    summary = {
        'total_exact_removed': total_exact,
        'total_semantic_removed': total_semantic,
        'total_synsets_processed': synset_count
    }
    with summary_out.open('w', encoding='utf-8') as fout_sum:
        fout_sum.write(json.dumps(summary, ensure_ascii=False) + '\n')

    logging.info(f"Processed {synset_count} synsets")
    logging.info(f"Totals – exact: {total_exact}, semantic: {total_semantic}")
    logging.info(f"Outputs in {out_dir}")


def main():
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)s [%(module)s] %(message)s',
        level=logging.INFO
    )
    inp = Path(args.input)
    if not inp.exists():
        logging.error(f"Input file not found: {inp}")
        raise FileNotFoundError(f"Input file not found: {inp}")
    model = load_model(args.model)
    filter_kb(inp, Path(args.out_dir), model,
              args.threshold, args.batch_size)


if __name__ == '__main__':
    main()
