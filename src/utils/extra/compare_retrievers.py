# run_retrieval.py

from datasets import load_dataset

from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb
from src.utils.data_utils import concatenate_question_choices
from src.datasets.dataset_loader import preprocess_dataset

def main():
    ckb_path = "data/ckb/cleaned/merged_filtered.jsonl"

    # we’ll compare both strategies
    strategies = ["retriever", "cner+retriever"]

    # load & preprocess the OBQA test set
    eval_dataset = load_dataset("allenai/openbookqa")["test"]
    eval_dataset = preprocess_dataset(eval_dataset, "obqa")
    id_to_sample = {ex["id"]: ex for ex in eval_dataset}

    # build one Retriever & one trained Retriever for each strategy
    retrievers = {}
    retrievers_trained = {}
    for strat in strategies:
        ckb = load_ckb(ckb_path, strat)
        retrievers[strat] = Retriever(
            model_name_or_path="intfloat/e5-base-v2",
            retrieval_strategy=strat,
            ckb=ckb,
            passage_prompt="passage: ",
            query_prompt="query: ",
        )
        retrievers_trained[strat] = Retriever(
            model_name_or_path="models/retriever_zebra/final",
            retrieval_strategy=strat,
            ckb=ckb,
            passage_prompt="passage: ",
            query_prompt="query: ",
        )

    while True:
        user_input = input("Enter sample ID or index (or press Enter to quit): ").strip()
        if not user_input:
            print("Exiting.")
            break

        # resolve input to a sample
        sample = id_to_sample.get(user_input)
        if sample is not None:
            idx = eval_dataset.data.index(sample) if sample in eval_dataset.data else "?"
        else:
            try:
                idx = int(user_input)
                sample = eval_dataset[idx]
            except (ValueError, IndexError):
                print("Invalid ID or index. Try again.")
                continue

        # build the actual text query from question + choices
        formatted_query = concatenate_question_choices(sample)

        print("\n" + "=" * 80)
        print(f"ID:     {sample['id']}")
        print(f"INDEX:  {idx}")
        print(f"QUERY:\n{formatted_query}")
        print(f"ANSWER: {sample['ground_truth']}\n")

        # for each strategy, show all four retrieval variants
        for strat in strategies:
            base_r = retrievers[strat]
            tr_r   = retrievers_trained[strat]

            print(f"--- STRATEGY: {strat.upper()} — BASE RETRIEVER ---")
            stmts = base_r.retrieve_top_k(
                formatted_query,
                top_k=20,
                diversify=False
            )
            print("\n".join(stmts), "\n")

            print(f"--- STRATEGY: {strat.upper()} — TRAINED RETRIEVER ---")
            stmts_tr = tr_r.retrieve_top_k(
                formatted_query,
                top_k=20,
                diversify=False
            )
            print("\n".join(stmts_tr), "\n")

            print(f"--- STRATEGY: {strat.upper()} — TRAINED RETRIEVER (MMR) ---")
            stmts_tr_mmr = tr_r.retrieve_top_k(
                formatted_query,
                top_k=20,
                diversify=True,
                re_rank="mmr",
                lambda_=0.8
            )
            print("\n".join(stmts_tr_mmr), "\n")

            print(f"--- STRATEGY: {strat.upper()} — TRAINED RETRIEVER (FILTER) ---")
            stmts_tr_filt = tr_r.retrieve_top_k(
                formatted_query,
                top_k=20,
                diversify=True,
                re_rank="filter",
                diversity_threshold=0.9
            )
            print("\n".join(stmts_tr_filt), "\n")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
