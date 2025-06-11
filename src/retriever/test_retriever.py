import glob
import os
import logging
from argparse import ArgumentParser
from typing import List

from settings.aliases import DATASET_NAME_TO_TAG, DATASET_TAG_TO_NAME
from src.datasets.dataset_loader import load_hf_dataset, preprocess_dataset
from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb, load_yaml
from src.utils.data_utils import concatenate_question_choices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def discover_models() -> List[str]:
    pattern = os.path.join("models", "retriever*", "final")
    found = sorted(glob.glob(pattern))
    valid = [p for p in found if os.path.isdir(p) or os.path.isfile(p)]
    return valid


def prompt_select(title: str, options: List[str]) -> int:
    print(f"\n🔸 {title}:")
    for idx, option in enumerate(options):
        print(f" [{idx}] {option}")
    while True:
        selection = input("Select number: ").strip()
        try:
            index = int(selection)
            if 0 <= index < len(options):
                return index
            else:
                print("❌ Invalid index.")
        except ValueError:
            print("❌ Please enter a number.")


def prompt_float(prompt_text: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    while True:
        value_str = input(f"{prompt_text} (float between {min_val} and {max_val}): ").strip()
        try:
            val = float(value_str)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"❌ Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("❌ Invalid number format.")


def run_retriever_tester(
    dataset_name: str,
    config_path: str,
    ckb_path: str,
    top_k: int,
):
    config = load_yaml(config_path)
    datasets_cache = {}

    retriever_models = discover_models()
    if not retriever_models:
        print("❌ No retriever models found under models/retriever*/final")
        return

    model_index = prompt_select("Select retriever model", retriever_models)
    current_model_path = retriever_models[model_index]

    retrieval_strategies = ["cner+retriever", "retriever"]
    retrieval_index = prompt_select("Select retrieval strategy", retrieval_strategies)
    current_retrieval_strategy = retrieval_strategies[retrieval_index]

    rerank_options = ["None", "mmr", "filter"]
    rerank_index = prompt_select("Select rerank type", rerank_options)
    current_rerank_type = None if rerank_options[rerank_index].lower() == "none" else rerank_options[rerank_index].lower()

    current_diversity_threshold = prompt_float("Set diversity threshold")
    current_lambda = prompt_float("Set lambda (balance factor)")

    print(f"🔎 Loading retriever: {current_model_path}")
    ckb = load_ckb(ckb_path, current_retrieval_strategy)
    retriever = Retriever(
        model_name_or_path=current_model_path,
        retrieval_strategy=current_retrieval_strategy,
        ckb=ckb,
        passage_prompt="passage: ",
        query_prompt="query: ",
    )

    current_dataset_name = dataset_name

    while True:
        command = input(
            "\nType sample ID, index, 'settings' to change settings, or 'q' to quit: "
        ).strip()

        if command.lower() in {"q", "quit", "exit"}:
            print("\n👋  Exiting… have a nice day!")
            break

        if command.lower() == "settings":
            print("\n🛠️  SETTINGS MENU")
            setting_options = ["Model", "Retrieval Strategy", "Rerank Type", "Diversity Threshold", "Lambda", "Cancel"]
            setting_index = prompt_select("Which setting do you want to change?", setting_options)

            if setting_options[setting_index] == "Model":
                model_index = prompt_select("Select retriever model", retriever_models)
                current_model_path = retriever_models[model_index]
                print(f"🔄 Reloading retriever: {current_model_path}")
                ckb = load_ckb(ckb_path, current_retrieval_strategy)
                retriever = Retriever(
                    model_name_or_path=current_model_path,
                    retrieval_strategy=current_retrieval_strategy,
                    ckb=ckb,
                    passage_prompt="passage: ",
                    query_prompt="query: ",
                )

            elif setting_options[setting_index] == "Retrieval Strategy":
                retrieval_index = prompt_select("Select retrieval strategy", retrieval_strategies)
                current_retrieval_strategy = retrieval_strategies[retrieval_index]
                print(f"🔄 Reloading retriever with new retrieval strategy: {current_retrieval_strategy}")
                ckb = load_ckb(ckb_path, current_retrieval_strategy)
                retriever = Retriever(
                    model_name_or_path=current_model_path,
                    retrieval_strategy=current_retrieval_strategy,
                    ckb=ckb,
                    passage_prompt="passage: ",
                    query_prompt="query: ",
                )

            elif setting_options[setting_index] == "Rerank Type":
                rerank_index = prompt_select("Select rerank type", rerank_options)
                current_rerank_type = None if rerank_options[rerank_index].lower() == "none" else rerank_options[rerank_index].lower()
                print(f"✅ Rerank type set to {current_rerank_type}")

            elif setting_options[setting_index] == "Diversity Threshold":
                current_diversity_threshold = prompt_float("Set diversity threshold")
                print(f"✅ Diversity threshold set to {current_diversity_threshold}")

            elif setting_options[setting_index] == "Lambda":
                current_lambda = prompt_float("Set lambda (balance factor)")
                print(f"✅ Lambda set to {current_lambda}")

            continue

        if current_dataset_name not in datasets_cache:
            print(f"📦 Loading dataset {current_dataset_name}")
            tag = DATASET_NAME_TO_TAG[current_dataset_name]
            raw = load_hf_dataset(config[tag])
            dataset = preprocess_dataset(raw, tag)
            datasets_cache[current_dataset_name] = dataset
        else:
            dataset = datasets_cache[current_dataset_name]

        try:
            idx = int(command)
            if 0 <= idx < len(dataset):
                sample_index = idx
            else:
                print("❌ Index out of range.")
                continue
        except ValueError:
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
            formatted_choices = "\n".join(
                [
                    f"{label}. {choice}"
                    for label, choice in zip(sample["choices"]["label"], sample["choices"]["text"])
                ]
            )
            print("Choices:\n" + formatted_choices)
        if "ground_truth" in sample:
            print(f'Answer: {sample["ground_truth"]}')

        retrieved_statements = retriever.retrieve_top_k(
            question_choices,
            top_k=top_k,
            pool_size=top_k * 2,
            re_rank=current_rerank_type,
            lambda_=current_lambda,
            diversity_threshold=current_diversity_threshold,
        )
        print(f"\n📚 Top-k (k={top_k}) from {current_model_path}")
        if retrieved_statements:
            for stmt in retrieved_statements:
                print(f" - {stmt}")
        else:
            print("  (No statements retrieved.)")
        print("=" * 80)


def main() -> None:
    parser = ArgumentParser(description="Interactive retriever tester (fully interactive version).")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ckb_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="settings/config.yaml")
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()
    args.dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)

    run_retriever_tester(
        dataset_name=args.dataset_name,
        config_path=args.config_path,
        ckb_path=args.ckb_path,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
