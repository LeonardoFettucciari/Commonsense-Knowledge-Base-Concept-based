from src.utils.io_utils import load_kb_as_dict


ckb = load_kb_as_dict("data/ckb/cleaned/ckb_data=wordnet|model=gemini-1.5-flash.jsonl")

synset_name = "electric.n.01"

synset_statements = ckb[synset_name]

print("hi")






























print("End")