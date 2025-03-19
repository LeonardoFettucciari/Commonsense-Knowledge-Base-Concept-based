from src.utils.model_utils import get_ner_pipeline
from src.utils.io_utils import load_kb_as_dict, load_yaml
from src.utils.data_utils import concatenate_question_choices, extract_unique_words, from_words_to_synsets, synsets_from_samples
import logging

from src.utils.io_utils import load_kb_as_dict
from src.retriever.retriever import Retriever
from src.utils.data_utils import synsets_from_samples
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


config = load_yaml("settings/config.yaml")


samples = ["An electric car runs on electricity via A. gasoline B. a power station C. electrical conductors D. fuel"]

synset_lists = synsets_from_samples(samples)

ckb = load_kb_as_dict("data/ckb/cleaned/ckb_data=wordnet|model=gemini-1.5-flash.jsonl")

ckb_statements = list(set([
            statement
            for synset_list in synset_lists
            for synset in synset_list
            for statement in ckb[synset.name()]
        ]))

logging.info(ckb_statements)

logging.info("Initializing retriever...")
retriever = Retriever(ckb_statements, config['retriever'], save_embeddings=False)

logging.info(f"Retrieveing {20} statements for evaluation dataset...")
eval_ckb_statements1 = retriever.retrieve(samples, 20)

eval_ckb_statements2 = retriever.retrieve(samples, 20)

logging.info(eval_ckb_statements1==eval_ckb_statements2)