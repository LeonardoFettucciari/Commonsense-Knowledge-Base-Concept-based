import json
import re
from tqdm import tqdm

# ── Compiled regex patterns ─────────────────────────────────────────────

PATTERN_THIS = re.compile(r'''
    ^This(?:\s+[\w-]+){1,4}\s+
    (grows|create|highlights|acts|adds|addresses|inspired|attracts|glows|loses|
    blooms|reproduces|increases|affects|commonly|specifically|falls|aimed|aims|
    also|allows|applies|arises|assists|assumes|became|benefits|broke|branches|
    can|catalyzes|causes|collects|commemorates|completes|conducts|distorts|
    connects|consists|contains|contributes|coordinates|creates|displays|
    disperses|disregarded|does|doesn['’]t|dries|ecompasses|emerged|emerges|
    emphasizes|employs|enables|encourages|encompasses|enhances|ensures|
    establishes|examines|excludes|exhibits|explains|extends|features|flows|
    follows|forms|fosters|gained|gathers|generates|governs|had|happens|has|
    helps|highlighted|illustrates|implies|includes|indicates|influences|informs|
    inhibits|instructs|interacts|involved|involves|is|lays|leads|lived|lowers|
    may|means|migrates|mimics|minimizes|moves|often|originated|originates|
    pairs|participates|performed|performs|persists|played|plays|prepares|
    prefers|primarily|produces|protects|provides|recognizes|evolved|followed|
    occurs|collaborates|laid|receives|spans|refers|recommends|reconstructs|
    relies|requires|resembles|results|retains|reveals|runs|saw|seeks|separates|
    serves|sets|shows|simplifies|stands|starts|suggest|suggests|supports|
    symbolized|symbolizes|targets|teaches|tends|thrives|this|took|transferred|
    transitioned|typically|undergoes|uses|values|was|weighed|works)\b
''', flags=re.IGNORECASE | re.VERBOSE)

PATTERN_THESE = re.compile(r'''
    ^These(?:\s+[\w-]+){1,4}\s+
    (will|might|grow|allow|build|create|were|emerged|toured|frequently|lived|primarily|convert|had|lack|highlight|act|add|address|inspire|attract|glow|
    lose|bloom|reproduce|increase|affect|commonly|specifically|fall|aim|also|hybernate|breathe| 
    allow|apply|arise|assist|assume|become|benefit|break|branch|can|thrived|feed|give|possess|catalyze|
    cause|collect|commemorate|complete|conduct|distort|connect|consist|contain|
    contribute|coordinate|display|disperse|disregard|do|don['’]t|dry|encompass|
    emerge|emphasize|employ|enable|encourage|enhance|ensure|establish|examine|walked|could|likely|usually|spend|
    exclude|exhibit|explain|extend|feature|flow|follow|form|foster|gain|gather|need|communicate|face|roost|secrete|
    generate|govern|have|happen|help|illustrate|imply|include|indicate|influence|
    inform|inhibit|instruct|interact|involve|are|lay|lead|live|lower|may|mean|
    migrate|mimic|minimize|move|often|originate|pair|participate|perform|persist|
    play|prepare|prefer|produce|protect|provide|recognize|evolve|occur|collaborate|
    receive|span|refer|recommend|reconstruct|rely|require|resemble|result|retain|
    reveal|run|see|seek|separate|serve|set|show|simplify|stand|start|suggest|
    support|symbolize|target|teach|tend|thrive|transition|typically|undergo|use|
    value|weigh|work)\b
''', flags=re.IGNORECASE | re.VERBOSE)

# ── Replacement function ───────────────────────────────────────────────

def replace_this_with_singular_lemma(entry, change_log):
    lemma = entry["synset_lemma"].replace("_", " ")
    lemma_cap = lemma[0].upper() + lemma[1:]

    updated_statements = []
    for s in entry["statements"]:
        if lemma.lower() in s.lower():
            updated_statements.append(s)
            continue

        # Match “These …”
        m2 = PATTERN_THESE.match(s)
        if m2:
            verb = m2.group(1)
            new_s = f"{lemma_cap} {verb}" + s[m2.end():]
            if new_s != s:
                change_log.append({"old_statement": s, "new_statement": new_s})
            updated_statements.append(new_s)
            continue

        # Match “This …”
        m1 = PATTERN_THIS.match(s)
        if m1:
            verb = m1.group(1)
            new_s = f"{lemma_cap} {verb}" + s[m1.end():]
            if new_s != s:
                change_log.append({"old_statement": s, "new_statement": new_s})
            updated_statements.append(new_s)
            continue

        # No match → keep original
        updated_statements.append(s)

    entry["statements"] = updated_statements
    return entry

# ── Main processing function ───────────────────────────────────────────

def process_jsonl(input_path, output_kb_path, output_log_path):
    change_log = []

    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_kb_path, "w", encoding="utf-8") as kb_outfile, \
         open(output_log_path, "w", encoding="utf-8") as log_outfile:

        for line in tqdm(infile, total=total_lines, desc="Processing statements"):
            if not line.strip():
                continue

            entry = json.loads(line)
            updated_entry = replace_this_with_singular_lemma(entry, change_log)
            json.dump(updated_entry, kb_outfile)
            kb_outfile.write("\n")

        for log_entry in change_log:
            json.dump(log_entry, log_outfile)
            log_outfile.write("\n")

# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_file = "data/ckb/cleaned/merged_filtered_this_ambiguous.jsonl"
    output_kb_file = "data/ckb/cleaned/merged_filtered.jsonl"
    output_log_file = "data/ckb/cleaned/merged_filtered_log.jsonl"
    process_jsonl(input_file, output_kb_file, output_log_file)
    print(f"✅ Updated KB saved to: {output_kb_file}")
    print(f"📝 Change log saved to: {output_log_file}")
