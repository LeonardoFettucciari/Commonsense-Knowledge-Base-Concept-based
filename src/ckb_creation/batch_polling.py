import argparse
import time
from pathlib import Path
from openai import OpenAI, OpenAIError
from loguru import logger

def upload_input_file(client, file_path: Path):
    try:
        with open(file_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        logger.info(f"Uploaded file: {file_path} → File ID: {uploaded.id}")
        return uploaded.id
    except OpenAIError as e:
        logger.error(f"Failed to upload {file_path}: {e}")
        return None

def submit_batch(client, file_id: str, description: str):
    try:
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )
        logger.info(f"Submitted batch {batch.id} for file ID {file_id}")
        return batch.id
    except OpenAIError as e:
        logger.error(f"Failed to create batch for file_id {file_id}: {e}")
        return None

def wait_for_completion(client, batch_id: str, poll_interval=30):
    logger.info(f"Waiting for batch {batch_id} to complete...")
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            logger.info(f"Batch {batch_id} status: {batch.status}")
            if batch.status in ["completed", "failed", "expired", "canceled"]:
                return batch
            time.sleep(poll_interval)
        except OpenAIError as e:
            logger.error(f"Error polling batch {batch_id}: {e}")
            time.sleep(poll_interval)

def download_results(client, batch, output_path: Path):
    file_id = batch.output_file_id
    if not file_id:
        logger.warning(f"No output file available for batch {batch.id}")
        return

    try:
        content_stream = client.files.content(file_id)
        with open(output_path, "wb") as f:
            for chunk in content_stream.iter_bytes():
                f.write(chunk)
        logger.success(f"Saved results to {output_path}")
    except OpenAIError as e:
        logger.error(f"Failed to download results for batch {batch.id}: {e}")

def process_batches(client, input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_file in input_dir.glob("*.jsonl"):
        logger.info(f"Processing file: {jsonl_file.name}")
        file_id = upload_input_file(client, jsonl_file)
        if not file_id:
            continue

        batch_id = submit_batch(client, file_id, description=f"Batch for {jsonl_file.name}")
        if not batch_id:
            continue

        batch = wait_for_completion(client, batch_id)
        output_path = output_dir / f"output_{jsonl_file.stem}.jsonl"
        download_results(client, batch, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI batch jobs from JSONL input files.")
    parser.add_argument("--input_dir", help="Directory containing .jsonl input files")
    parser.add_argument("--output_dir", help="Directory to write output result files")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    args = parser.parse_args()

    # Configure loguru
    logger.add("batch_runner.log", rotation="1 MB", level="INFO", backtrace=True, diagnose=True)
    logger.info("Starting batch processing...")

    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key)
    process_batches(client, Path(args.input_dir), Path(args.output_dir))
