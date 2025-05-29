import os
import shutil
from pathlib import Path


def find_latest_subdir(path):
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.name)


def copy_accuracy_tsv_files(
    source_root: Path, output_root: Path, run_names: list[str], datasets: list[str], models: list[str]
):
    for dataset in datasets:
        for model in models:
            for run_name in run_names:
                source_run_path = source_root / dataset / model / run_name
                if not source_run_path.exists():
                    continue

                latest_date_folder = find_latest_subdir(source_run_path)
                if latest_date_folder is None:
                    continue

                accuracy_path = latest_date_folder / "accuracy"
                if not accuracy_path.exists():
                    continue

                # Define the destination path
                dest_path = output_root / dataset / model / run_name / latest_date_folder.name
                dest_path.mkdir(parents=True, exist_ok=True)

                # Copy .tsv files only
                for file in accuracy_path.glob("*.tsv"):
                    shutil.copy(file, dest_path / file.name)
                    print(f"Copied {file} to {dest_path / file.name}")


if __name__ == "__main__":
    SOURCE_ROOT = Path("outputs/inference")
    OUTPUT_ROOT = Path("outputs/download")
    RUN_NAMES = ["trained_retriever_mmr_fewshot"]
    DATASETS = ["csqa", "obqa", "qasc"]
    MODELS = [
        "Llama-3.1-8B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]

    copy_accuracy_tsv_files(SOURCE_ROOT, OUTPUT_ROOT, RUN_NAMES, DATASETS, MODELS)
