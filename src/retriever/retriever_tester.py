import logging
from argparse import ArgumentParser
from typing import List

from settings.aliases import DATASET_NAME_TO_TAG, DATASET_TAG_TO_NAME
from src.datasets.dataset_loader import load_hf_dataset, preprocess_dataset
from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb, load_yaml
from src.utils.data_utils import concatenate_question_choices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _prompt_for_index_or_id(dataset: list[dict]) -> int | None:
    dataset_size = len(dataset)
    while True:
        raw = input(f"\nEnter a sample index (0 to {dataset_size - 1}) or sample ID (or 'q' to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        '''
        try:
            idx = int(raw)
            if 0 <= idx < dataset_size:
                return idx
            else:
                print("🔍  No sample at that index. Trying as ID...")
        except ValueError:
            pass
        '''
        for i, sample in enumerate(dataset):
            if str(sample.get("id")) == raw:
                return i

        print("❌  No sample found with that index or ID.")


def run_retriever_test(
    dataset_name: str,
    config_path: str,
    retrieval_strategy: str,
    ckb_path: str,
    retriever_model: str,
    top_k: int,
    rerank_type: str | None,
    lambda_: float,
    diversity_threshold: float,
) -> None:
    logging.info("Loading configuration from %s", config_path)
    config = load_yaml(config_path)

    dataset_tag = DATASET_NAME_TO_TAG[dataset_name]
    logging.info("Loading dataset %s", dataset_name)
    eval_dataset = load_hf_dataset(config[dataset_tag])
    eval_dataset = preprocess_dataset(eval_dataset, dataset_tag)
    logging.info("Dataset loaded (size=%d)", len(eval_dataset))

    logging.info("Loading CKB from %s", ckb_path)
    ckb = load_ckb(ckb_path, retrieval_strategy)

    retriever = Retriever(
        model_name_or_path=retriever_model,
        retrieval_strategy=retrieval_strategy,
        ckb=ckb,
        passage_prompt="passage: ",
        query_prompt="query: ",
    )

    while True:
        index = _prompt_for_index_or_id(eval_dataset)
        if index is None:
            print("\n👋  Exiting… have a nice day!")
            break

        sample = eval_dataset[index]
        question_choices = concatenate_question_choices(sample)

        print("\n" + "=" * 80)
        print(f"📄  SAMPLE {index}")
        print("-" * 80)
        print(f"Question: {sample.get('question', 'N/A')}")
        if "choices" in sample:
            formatted_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])
            print("Choices:\n" + formatted_choices)
        if "ground_truth" in sample:
            print(f"Answer: {sample['ground_truth']}")

        retrieved_statements: List[str] = retriever.retrieve_top_k(
            question_choices,
            top_k=top_k,
            pool_size=top_k * 2,
            re_rank=rerank_type,
            lambda_=lambda_,
            diversity_threshold=diversity_threshold,
        )

        print("\n📚 Top‑k retrieved knowledge statements:")
        if retrieved_statements:
            for stmt in retrieved_statements:
                print(f"{stmt}")
        else:
            print("  (No statements retrieved.)")
        print("=" * 80)


def main() -> None:
    parser = ArgumentParser(description="Interactive retriever-only tester.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ckb_path", type=str, required=True)
    parser.add_argument("--retrieval_strategy", type=str, default="bm25")
    parser.add_argument("--config_path", type=str, default="settings/config.yaml")
    parser.add_argument("--retriever_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--lambda_", type=float, default=0.0)
    parser.add_argument("--rerank_type", type=str, default=None)
    parser.add_argument("--diversity_threshold", type=float, default=0.7)

    args = parser.parse_args()

    args.dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)
    args.rerank_type = args.rerank_type if args.rerank_type != "" else None
    run_retriever_test(**vars(args))


if __name__ == "__main__":
    main()
