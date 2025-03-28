from argparse import ArgumentParser
import json
import csv
import datetime
import torch
import transformers

# -------------------------------------------------------------------------
#  Vera class (unchanged from your second snippet, just renamed)
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
#  Filter & Statistics, producing TWO separate TSVs
# -------------------------------------------------------------------------

def filter_and_report_two_tsvs(
    kb_jsonl_path,
    per_synset_tsv_path,
    global_tsv_path,
    threshold=0.5
):
    """
    Reads a JSONL file where each line is a dict like:
        {
            "synset": "some_id",
            "statements": ["stmt1", "stmt2", ...]
        }
    For each synset:
      - runs Vera
      - splits into removed vs kept statements
      - writes a row in per_synset_tsv_path with:
         synset_id, pct_removed, removed_list, removed_count, kept_list, kept_count

    Then writes one row to global_tsv_path with:
        total_statements, removed_statements, kept_statements, pct_removed
    """
    vera = Vera()

    # Keep track of overall counts across all synsets
    global_total = 0
    global_removed = 0
    global_kept = 0

    # Open the per-synset stats TSV
    with open(per_synset_tsv_path, 'w', newline='', encoding='utf-8') as outf_synset:
        writer_synset = csv.writer(outf_synset, delimiter='\t')
        # header for the per-synset file
        writer_synset.writerow([
            "synset_id",
            "pct_removed",
            "removed_list",
            "removed_count",
            "kept_list",
            "kept_count"
        ])

        # Read JSONL line-by-line
        with open(kb_jsonl_path, 'r', encoding='utf-8') as inf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                synset_id = data.get("synset") or data.get("synset_id")
                statements = data.get("statements", [])

                if not statements:
                    # Write an empty row if no statements
                    writer_synset.writerow([synset_id, 0, "", 0, "", 0])
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
                pct_removed = round(100.0 * removed_count / total_count, 2)  # e.g. 33.33
                # join lists as strings
                removed_str = "; ".join(removed_statements)
                kept_str = "; ".join(kept_statements)

                # 4) write row
                writer_synset.writerow([
                    synset_id,
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

    # After finishing all synsets, write the global stats to a separate TSV
    if global_total > 0:
        global_pct_removed = round(100.0 * global_removed / global_total, 2)
    else:
        global_pct_removed = 0.0

    # Write to global TSV
    with open(global_tsv_path, 'w', newline='', encoding='utf-8') as outf_global:
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
#  Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    kb_jsonl_path = "kb.jsonl"
    per_synset_tsv_path = "per_synset_stats.tsv"
    global_tsv_path = "global_stats.tsv"

    """
    Parse arguments and launch the inference procedure.
    """
    parser = ArgumentParser(description="Merging script for two CKB json files.")
    parser.add_argument(
        "--kb_jsonl_path",
        type=str,
        required=True,
        help="Path to the ckb as jsonl file."
    )
    parser.add_argument(
        "--threshold",
        type=str,
        required=True,
        help="Source CKB path."
    )
    
    args = parser.parse_args()
    
    filter_and_report_two_tsvs(**vars(args))
