import os
import shutil
from pathlib import Path
import argparse

def find_latest_subdir(path):
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.name)

def copy_accuracy_tsv_files(
    source_root: Path,
    output_root: Path,
    run_names: list[str],
    datasets: list[str],
    models: list[str],
    use_accuracy_folder: bool = True,
):
    for dataset in datasets:
        for model in models:
            for run_name in run_names:
                source_run_path = source_root / dataset / model / run_name
                if not source_run_path.exists():
                    print(f"Skipping missing path: {source_run_path}")
                    continue

                # Check if subfolders exist
                latest_date_folder = find_latest_subdir(source_run_path)
                if latest_date_folder is None:
                    # No subfolders → use source_run_path itself
                    selected_source_folder = source_run_path
                    selected_folder_name = "root"
                else:
                    selected_source_folder = latest_date_folder
                    selected_folder_name = latest_date_folder.name

                # Decide source path
                if use_accuracy_folder:
                    source_tsv_path = selected_source_folder / "accuracy"
                else:
                    source_tsv_path = selected_source_folder

                if not source_tsv_path.exists():
                    print(f"Skipping: path does not exist: {source_tsv_path}")
                    continue

                # Define the destination path
                dest_path = output_root / dataset / model / run_name / selected_folder_name
                dest_path.mkdir(parents=True, exist_ok=True)

                # Copy .tsv files only
                tsv_files = list(source_tsv_path.glob("*.tsv"))
                if not tsv_files:
                    print(f"No TSV files found in {source_tsv_path}")
                    continue

                for file in tsv_files:
                    shutil.copy(file, dest_path / file.name)
                    print(f"✅ Copied {file} → {dest_path / file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_names",
        nargs="+",
        required=True,
        help="List of run names to copy (e.g. --run_names run1 run2 run3)"
    )
    parser.add_argument(
        "--no_accuracy",
        action="store_true",
        help="If set, DO NOT copy from 'accuracy/' folder (copy from latest run folder instead)."
    )
    args = parser.parse_args()

    SOURCE_ROOT = Path("outputs/inference")
    OUTPUT_ROOT = Path("outputs/upload2gdrive")
    DATASETS = ["csqa", "obqa", "qasc"]
    MODELS = [
        "Llama-3.1-8B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]

    copy_accuracy_tsv_files(
        SOURCE_ROOT,
        OUTPUT_ROOT,
        args.run_names,
        DATASETS,
        MODELS,
        use_accuracy_folder=not args.no_accuracy,
    )
