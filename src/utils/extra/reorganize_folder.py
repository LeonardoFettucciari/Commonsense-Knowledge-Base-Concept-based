#!/usr/bin/env python3
"""
Reorganise a *Folder 1* directory **in place** so it matches the *Folder 2*
structure.

Baseline rules (finalised)
-------------------------
A file is routed to the **`baselines/`** experiment folder when **any** of the
following holds ‑ anywhere in the tree (top‑level or under `accuracy/`):

* The filename **has no `retrieval_strategy=` field** at all.
* `model=Llama-3.2-3B-Instruct` appears in its bar‑separated filename.
* The `prompt=` value equals **one of**
  `zeroshot`, `zeroshot_cot`, `fewshot`, `fewshot_cot`, **or their short
  tags** `zs`, `zscot`, `fs`, `fscot` (trailing underscore allowed).
* The literal word "baseline" appears in the filename.

These rules now catch cases like
`xFinder_accuracy|model=Llama-3.2-3B-Instruct.jsonl` and ensure they land in
`baselines/accuracy/`.

If none of the baseline conditions match **and** the filename *does* specify a
`retrieval_strategy=…`, we map that value using `EXPERIMENT_NAME_FROM_RETRIEVAL_STRATEGY`.
Unknown strategies fall back to `misc/`.

Usage
-----
    python reorganize_experiments.py /path/to/Folder1 --dry-run  # preview
    python reorganize_experiments.py /path/to/Folder1            # apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENT_NAME_FROM_RETRIEVAL_STRATEGY: dict[str, str] = {
    "full_ckb": "retriever",
    "cner_filter": "cner",
    "retriever": "retriever",
}

# Baseline indicators --------------------------------------------------------
BASELINE_MODELS = {
    "Llama-3.2-3B-Instruct",
}
# Long prompt names
BASELINE_PROMPTS = {
    "zeroshot",
    "zeroshot_cot",
    "fewshot",
    "fewshot_cot",
}
# Short tags found especially under accuracy/
BASELINE_PROMPT_SHORT_TAGS = {
    "zs": "zeroshot",
    "zscot": "zeroshot_cot",
    "fs": "fewshot",
    "fscot": "fewshot_cot",
}
# Unified lookup set for quick membership tests
_ALL_BASELINE_PROMPTS = BASELINE_PROMPTS | set(BASELINE_PROMPT_SHORT_TAGS)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def _value_from_bar_separated(name: str, key: str) -> str | None:
    """Return the value for *key* in a *foo|bar=baz|qux* style filename."""
    for segment in name.split("|"):
        if segment.startswith(f"{key}="):
            return segment.split("=", 1)[1].split(".")[0]  # strip extension
    return None


def _has_field(name: str, key: str) -> bool:
    """True if *key=* appears as a bar‑separated segment in *name*."""
    return any(seg.startswith(f"{key}=") for seg in name.split("|"))


def _is_baseline_filename(name: str) -> bool:
    """Return True if filename triggers baseline rules by itself."""
    # 0. No retrieval_strategy field → baseline
    if not _has_field(name, "retrieval_strategy"):
        return True

    # 1. Model check -------------------------------------------------------
    model_val = _value_from_bar_separated(name, "model")
    if model_val in BASELINE_MODELS:
        return True

    # 2. Prompt check ------------------------------------------------------
    prompt_val = _value_from_bar_separated(name, "prompt")
    if prompt_val and prompt_val.rstrip("_") in _ALL_BASELINE_PROMPTS:
        return True

    # 3. Legacy keyword ----------------------------------------------------
    return "baseline" in name.lower()


def _is_baseline_prompt_folder(folder_name: str) -> bool:
    """True if a directory component (e.g. accuracy/zs/) signals baseline."""
    return folder_name.rstrip("_") in _ALL_BASELINE_PROMPTS

# ----------------------------------------------------------------------------
# Experiment‑resolution helpers
# ----------------------------------------------------------------------------

def _experiment_from_accuracy_path(rel: Path) -> str:
    """Determine experiment folder for a path under the old *accuracy/* tree."""
    # Nested path: accuracy/<first>/<...>
    if len(rel.parts) > 1:
        first = rel.parts[0]
        if _is_baseline_prompt_folder(first):
            return "baselines"
        return first  # Explicit experiment folder given

    # Flat accuracy file → decide from filename
    fname = rel.name
    if _is_baseline_filename(fname):
        return "baselines"

    strat = _value_from_bar_separated(fname, "retrieval_strategy")
    if strat:
        return EXPERIMENT_NAME_FROM_RETRIEVAL_STRATEGY.get(strat, strat)
    # If we reach here, strat is None but baseline rules did not match (shouldn't happen)
    return "misc"


def _experiment_from_top_level_file(name: str) -> str:
    """Determine experiment folder for a top‑level result file in old layout."""
    if _is_baseline_filename(name):
        return "baselines"

    strat = _value_from_bar_separated(name, "retrieval_strategy")
    if strat:
        return EXPERIMENT_NAME_FROM_RETRIEVAL_STRATEGY.get(strat, strat)

    # No retrieval_strategy and no baseline rule (shouldn't happen)
    return "misc"

# ----------------------------------------------------------------------------
# Core mover
# ----------------------------------------------------------------------------

def _plan_moves(folder: Path) -> List[Tuple[Path, Path]]:
    """Return a list of (src, dest) moves needed to achieve the new layout."""
    moves: list[tuple[Path, Path]] = []
    accuracy_root = folder / "accuracy"

    # 1. Top‑level files (main results) -------------------------------------
    for item in folder.iterdir():
        if item.is_file():
            exp = _experiment_from_top_level_file(item.name)
            dest = folder / exp / item.name
            if dest != item:
                moves.append((item, dest))

    # 2. Files inside the old accuracy/ tree --------------------------------
    if accuracy_root.exists():
        for file_path in accuracy_root.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(accuracy_root)
            exp = _experiment_from_accuracy_path(rel)
            sub_rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)
            dest = folder / exp / "accuracy" / sub_rel
            if dest != file_path:
                moves.append((file_path, dest))

    return moves


def _apply_moves(moves: List[Tuple[Path, Path]], dry_run: bool) -> None:
    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"[DRY‑RUN] {src.relative_to(src.anchor)} -> {dest.relative_to(dest.anchor)}")
        else:
            shutil.move(str(src), dest)

# ----------------------------------------------------------------------------
# Public entry‑points
# ----------------------------------------------------------------------------

def reorganise_in_place(folder: Path, dry_run: bool = False) -> None:
    """Transform *folder* from *Folder 1* layout into *Folder 2* layout in place."""
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory")

    moves = _plan_moves(folder)
    if not moves:
        print("Nothing to move – this directory already looks like Folder 2.")
        return

    _apply_moves(moves, dry_run)
    summary = "Planned" if dry_run else "Moved"
    print(f"{summary} {len(moves)} files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorganise Folder 1 layout *in place* so it matches Folder 2.")
    parser.add_argument("folder", help="Path to the Folder 1 directory (will be modified in place)")
    parser.add_argument("--dry-run", action="store_true", help="Show planned moves without making changes")
    args = parser.parse_args()

    reorganise_in_place(Path(args.folder), args.dry_run)


if __name__ == "__main__":
    main()
