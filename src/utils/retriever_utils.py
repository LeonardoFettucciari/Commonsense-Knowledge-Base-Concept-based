from src.utils.data_utils import concatenate_question_choices, synsets_from_samples
import logging

def retrieve_top_k_statements(retriever, sample, ckb, k, retrieval_scope):

    if retrieval_scope == "cner_synset_filtered_kb": # CKB here is a dict synset:statements
        # Concatenate question + choices
        formatted_question = concatenate_question_choices(sample)
        # Extract synsets from samples
        sample_synsets = synsets_from_samples(formatted_question)
        # Merge statements of such synsets from ckb dict
        ckb_statements = list(set([
            statement
            for synset in sample_synsets
            for statement in ckb[synset.name()]
        ]))
        # Set retriever's passages
        retriever.set_passages(ckb_statements)
        # Retrieve top k statements
        return retriever.retrieve(formatted_question, k)
    
    elif retrieval_scope == "full_ckb": # CKB here is a list of all ckb statements
        # Concatenate question + choices
        formatted_question = concatenate_question_choices(sample)
        # Retrieve top k statements
        return retriever.retrieve(formatted_question, k)
    else:
        ValueError(f"Retrieval scope not supported: {retrieval_scope}.")


    

def add_ckb_statements_to_samples(samples, ckb_statements_list):
    samples = [samples] if not isinstance(samples, list) else samples
    ckb_statements_list = [ckb_statements_list] if not isinstance(ckb_statements_list[0], list) else ckb_statements_list

    if len(samples) != len(ckb_statements_list):
        logging.warning(f"Mismatch: {len(samples)} samples but {len(ckb_statements_list)} ckb_statements.")

    for s, statements in zip(samples, ckb_statements_list):
        s["ckb_statements"] = statements or []
