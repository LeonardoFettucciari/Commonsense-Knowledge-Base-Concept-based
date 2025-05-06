import os
import shutil
from pathlib import Path

def process_directory(input_dir, output_root):
    input_dir = Path(input_dir).resolve()
    output_root = Path(output_root).resolve()
    output_dir = output_root / input_dir.name

    # Copy the entire directory tree to the output location
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    # Walk the copied output directory
    for dirpath, dirnames, filenames in os.walk(output_dir, topdown=True):
        if 'accuracy' in dirnames:
            accuracy_path = Path(dirpath) / 'accuracy'
            parent_dir = Path(dirpath)

            # Remove all files in the parent directory
            for file in parent_dir.iterdir():
                if file.is_file():
                    file.unlink()

            # Move all contents of accuracy folder to the parent directory
            for item in accuracy_path.iterdir():
                shutil.move(str(item), str(parent_dir))

            # Remove the empty accuracy folder
            accuracy_path.rmdir()

            # Remove .jsonl files in the parent directory (if any)
            for jsonl_file in parent_dir.glob("*.jsonl"):
                jsonl_file.unlink()

            # Prevent further descent into removed accuracy folder
            dirnames.remove('accuracy')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process accuracy directories.")
    parser.add_argument("input_dir", help="Path to the input directory")
    parser.add_argument("--output_dir", default="outputs/download", help="Path to the output root directory")

    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
