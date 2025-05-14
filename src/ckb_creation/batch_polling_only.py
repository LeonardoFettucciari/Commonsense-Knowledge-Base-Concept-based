import argparse
import time
from pathlib import Path
from openai import OpenAI, OpenAIError
from loguru import logger


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI batch jobs from JSONL input files.")
    parser.add_argument("--output_dir", help="Directory to write output result files")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    args = parser.parse_args()

    # Configure loguru
    logger.add("batch_runner.log", rotation="1 MB", level="INFO", backtrace=True, diagnose=True)
    logger.info("Starting batch processing...")

    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key)
    batch = wait_for_completion(client, "batch_68246f3809208190a8f96f39be1e3b67")
    output_path = Path(args.output_dir) / f"output_test.jsonl"
    download_results(client, batch, output_path)
