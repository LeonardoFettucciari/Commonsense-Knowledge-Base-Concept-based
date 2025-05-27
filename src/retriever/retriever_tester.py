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
    config = load_yaml(config_path)

    datasets_cache = {}
    retriever = None
    current_dataset_name = dataset_name
    current_rerank = rerank_type

    while True:
        command = input(f"\nType sample ID, index, 'set dataset <name>', 'set rerank <type>', or 'q' to quit: ").strip()
        
        if command.lower() in {"q", "quit", "exit"}:
            print("\n👋  Exiting… have a nice day!")
            break

        if command.startswith("set dataset "):
            new_name = command[len("set dataset "):].strip()
            new_name = DATASET_TAG_TO_NAME.get(new_name, new_name)
            if new_name not in DATASET_NAME_TO_TAG:
                print(f"❌ Unknown dataset '{new_name}'. Available: {', '.join(DATASET_NAME_TO_TAG.keys())}")
                continue
            current_dataset_name = new_name
            print(f"✅ Dataset switched to {current_dataset_name}")
            continue

        if command.startswith("set rerank "):
            new_rerank = command[len("set rerank "):].strip().lower()
            if new_rerank not in {"mmr", "filter", "none"}:
                print("❌ Rerank type must be one of: mmr, filter, none")
                continue
            current_rerank = None if new_rerank == "none" else new_rerank
            print(f"✅ Rerank type set to {current_rerank}")
            continue

        # Load dataset if needed
        if current_dataset_name not in datasets_cache:
            print(f"📦 Loading dataset {current_dataset_name}")
            tag = DATASET_NAME_TO_TAG[current_dataset_name]
            raw = load_hf_dataset(config[tag])
            dataset = preprocess_dataset(raw, tag)
            datasets_cache[current_dataset_name] = dataset
        else:
            dataset = datasets_cache[current_dataset_name]

        # Init retriever if needed
        if retriever is None:
            print(f"🔎 Loading retriever for strategy {retrieval_strategy}")
            ckb = load_ckb(ckb_path, retrieval_strategy)
            retriever = Retriever(
                model_name_or_path=retriever_model,
                retrieval_strategy=retrieval_strategy,
                ckb=ckb,
                passage_prompt="passage: ",
                query_prompt="query: ",
            )

        # Try parsing as index or ID
        '''
        try:
            idx = int(command)
            if 0 <= idx < len(dataset):
                sample_index = idx
            else:
                print("❌ Index out of range.")
                continue
        except ValueError:
        '''
        matches = [i for i, s in enumerate(dataset) if str(s.get("id")) == command]
        if not matches:
            print("❌ No sample found with that ID.")
            continue
        sample_index = matches[0]

        sample = dataset[sample_index]
        question_choices = concatenate_question_choices(sample)

        print("\n" + "=" * 80)
        print(f"📄  SAMPLE {sample_index}")
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
            re_rank=current_rerank,
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
