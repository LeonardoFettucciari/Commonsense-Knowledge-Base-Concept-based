from argparse import ArgumentParser
import json
import csv
import os
import datetime
import torch
import transformers
from tqdm import tqdm

# -------------------------------------------------------------------------
#  Vera class
# -------------------------------------------------------------------------
MODEL_NAME = 'liujch1998/vera'
MODE = 'normal'  # or 'debug'
HF_TOKEN_DOWNLOAD = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vera:
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_auth_token=HF_TOKEN_DOWNLOAD
        )
        if MODE == 'debug':
            return
        self.model = transformers.T5EncoderModel.from_pretrained(
            MODEL_NAME,
            use_auth_token=HF_TOKEN_DOWNLOAD,
            low_cpu_mem_usage=True,
            device_map='auto',
            torch_dtype='auto',
            offload_folder='offload'
        )
        self.model.D = self.model.shared.embedding_dim
        self.linear = torch.nn.Linear(self.model.D, 1, dtype=self.model.dtype).to(device)
        self.linear.weight = torch.nn.Parameter(self.model.shared.weight[32099, :].unsqueeze(0))
        self.linear.bias = torch.nn.Parameter(self.model.shared.weight[32098, 0].unsqueeze(0))
        self.model.eval()
        self.t = self.model.shared.weight[32097, 0].item()

    def runs(self, statements):
        """
        Batch process a list of statements.
        Return a list of dicts, each containing:
            'statement', 'logit', 'logit_calibrated', 'score', 'score_calibrated'
        """
        if MODE == 'debug':
            return [{
                'timestamp': datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                'statement': st,
                'logit': 0.0,
                'logit_calibrated': 0.0,
                'score': 0.5,
                'score_calibrated': 0.5,
            } for st in statements]

        tok = self.tokenizer.batch_encode_plus(
            statements,
            return_tensors='pt',
            padding='longest',
            truncation='longest_first',
            max_length=128
        )
        input_ids = tok.input_ids.to(device)
        attention_mask = tok.attention_mask.to(device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state  # (B, L, D)

            # Identify last valid token for each sequence
            last_indices = attention_mask.sum(dim=1, keepdim=True) - 1  # shape (B,1)
            last_indices = last_indices.unsqueeze(-1).expand(-1, -1, self.model.D)  # (B,1,D)

            hidden = last_hidden_state.gather(dim=1, index=last_indices).squeeze(1)  # (B, D)
            logits = self.linear(hidden).squeeze(-1)                                # (B)
            logits_calibrated = logits / self.t                                     # (B)
            scores = torch.sigmoid(logits)                                          # (B)
            scores_calibrated = torch.sigmoid(logits_calibrated)                    # (B)

        results = []
        for st, l, lc, sc, scc in zip(statements, logits, logits_calibrated, scores, scores_calibrated):
            results.append({
                'timestamp': datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                'statement': st,
                'logit': l.item(),
                'logit_calibrated': lc.item(),
                'score': sc.item(),
                'score_calibrated': scc.item(),
            })
        return results


# -------------------------------------------------------------------------
#  Filter & Statistics, producing two separate TSVs
# -------------------------------------------------------------------------
def filter_and_stats(
    kb_jsonl_path,
    per_synset_tsv_filename,
    global_tsv_filename,
    threshold,
):
    """
    Reads a JSONL file where each line is a dict like:
        {
            "synset_name": "some_id",
            "statements": ["stmt1", "stmt2", ...]
        }
    For each synset:
      - runs Vera
      - splits into removed vs kept statements
      - writes a row in per_synset_tsv_path with:
         synset_name, pct_removed, removed_list, removed_count, kept_list, kept_count

    Then writes one row to global_tsv_path with:
        total_statements, removed_statements, kept_statements, pct_removed

    Additionally, writes a new filtered JSONL to `vera_<original_file_name>` which
    has the same structure as the input but only with the kept statements.
    """
    output_dir = os.path.dirname(kb_jsonl_path)
    vera = Vera()

    # Prepare name for the new filtered ckb file.
    original_filename = os.path.basename(kb_jsonl_path)  # e.g. 'original_input_name.jsonl'
    filtered_ckb_filename = f"vera_{original_filename}"   # e.g. 'vera_original_input_name.jsonl'
    filtered_ckb_path = os.path.join(output_dir, filtered_ckb_filename)

    # First, count how many lines are in the input file so tqdm can track total progress
    total_lines = sum(1 for _ in open(kb_jsonl_path, 'r', encoding='utf-8'))

    # Keep track of overall counts across all synsets
    global_total = 0
    global_removed = 0
    global_kept = 0

    # Open the per-synset stats TSV
    with open(os.path.join(output_dir, per_synset_tsv_filename), 'w', newline='', encoding='utf-8') as outf_synset, \
         open(filtered_ckb_path, 'w', encoding='utf-8') as outf_ckb, \
         open(kb_jsonl_path, 'r', encoding='utf-8') as inf, \
         tqdm(total=total_lines, desc="Processing Synsets") as pbar:

        writer_synset = csv.writer(outf_synset, delimiter='\t')
        # header for the per-synset file
        writer_synset.writerow([
            "synset_name",
            "pct_removed",
            "removed_list",
            "removed_count",
            "kept_list",
            "kept_count"
        ])

        # Read JSONL line-by-line
        for line in inf:
            pbar.update(1)  # Update the progress bar every time we read a line

            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            synset_name = data.get("synset_name")
            statements = data.get("statements", [])

            if not statements:
                # Write an empty row if no statements
                writer_synset.writerow([synset_name, 0, "", 0, "", 0])
                # Write a JSON line with zero statements to the filtered file
                data["statements"] = []
                outf_ckb.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # 1) get Vera results
            results = vera.runs(statements)

            # 2) filter
            removed_statements = []
            kept_statements = []
            for r in results:
                if r['score_calibrated'] < threshold:
                    removed_statements.append(r['statement'])
                else:
                    kept_statements.append(r['statement'])

            # 3) stats for this synset
            total_count = len(statements)
            removed_count = len(removed_statements)
            kept_count = len(kept_statements)
            pct_removed = round(100.0 * removed_count / total_count, 2)

            # join lists as strings for the per-synset TSV
            removed_str = "\n".join(removed_statements)
            kept_str = "\n".join(kept_statements)

            # 4) write row to the per-synset TSV
            writer_synset.writerow([
                synset_name,
                pct_removed,
                removed_str,
                removed_count,
                kept_str,
                kept_count
            ])

            # 5) update global counters
            global_total += total_count
            global_removed += removed_count
            global_kept += kept_count

            # 6) Write a new JSON line to the filtered ckb, with only kept statements
            data["statements"] = kept_statements
            outf_ckb.write(json.dumps(data, ensure_ascii=False) + "\n")

    # After finishing all synsets, write the global stats to a separate TSV
    if global_total > 0:
        global_pct_removed = round(100.0 * global_removed / global_total, 2)
    else:
        global_pct_removed = 0.0

    # Write to global TSV
    with open(os.path.join(output_dir, global_tsv_filename), 'w', newline='', encoding='utf-8') as outf_global:
        writer_global = csv.writer(outf_global, delimiter='\t')
        # header
        writer_global.writerow([
            "total_statements",
            "removed_statements",
            "kept_statements",
            "pct_removed"
        ])
        # single row of global stats
        writer_global.writerow([
            global_total,
            global_removed,
            global_kept,
            global_pct_removed
        ])


# -------------------------------------------------------------------------
#  Arguments
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="Merging script for two CKB json files.")
    parser.add_argument(
        "--kb_jsonl_path",
        type=str,
        required=True,
        help="Path to the ckb as jsonl file."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.5,
        help="Threshold for filtering statements."
    )
    parser.add_argument(
        "--per_synset_tsv_filename",
        type=str,
        required=False,
        default="per_synset_stats.tsv",
        help="Filename of per-synset stats."
    )
    parser.add_argument(
        "--global_tsv_filename",
        type=str,
        required=False,
        default="global_stats.tsv",
        help="Filename of global stats."
    )
    
    args = parser.parse_args()
    filter_and_stats(**vars(args))
