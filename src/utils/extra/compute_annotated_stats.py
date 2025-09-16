import csv
from collections import Counter

csv_path = "outputs/annotations/stats/stats.csv"   # ← adjust if needed

# --------------------------------------------
# 1) Basic counters you already had
broken_count = 0
count_relevant_factual_helpful = 0
count_relevant_factual_harmful = 0
count_relevant_factual_neutral = 0
valid_sample_count = 0

# --------------------------------------------
# 2) New: distribution of the two-tick case flags
#    Keys are the exact two-glyph strings, e.g. "✅✅"
case_distribution_neutral = Counter({"✅✅": 0, "❌✅": 0, "✅❌": 0, "❌❌": 0})

# --------------------------------------------
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        valid_status = row.get("Valid question?", "")
        relevant     = row.get("Relevant?", "")
        factual      = row.get("Factual?", "")
        helpfulness  = row.get("Helpful?", "")

        # ─── skip broken questions ───────────────────────────────────────────
        if valid_status == "Broken":
            broken_count += 1
            continue
        valid_sample_count += 1

        # ─── your original tallies ───────────────────────────────────────────
        if relevant == "Relevant" and factual == "Factual":
            if helpfulness == "Helpful":
                count_relevant_factual_helpful += 1
            elif helpfulness == "Harmful":
                count_relevant_factual_harmful += 1
            elif helpfulness == "Neutral":
                count_relevant_factual_neutral += 1

        # ─── NEW: neutral-only case distribution ─────────────────────────────
        if helpfulness == "Neutral":
            # OPTION A – one column already contains the two glyphs
            case_string = row.get("case", "")            # column name 'case'

            # OPTION B – two separate columns, e.g. 'Case 1' & 'Case 2'
            # case_string = f"{row.get('Case 1', '')}{row.get('Case 2', '')}"

            if case_string in case_distribution_neutral:
                case_distribution_neutral[case_string] += 1
            else:
                # In case there are unexpected values, collect them too
                case_distribution_neutral[case_string] += 1

# --------------------------------------------
# 3) Print the results
def fmt(count, total):           # helper for n (p%)
    return f"{count} ({(count/total*100):.2f}%)" if total else f"{count} (0.00%)"

print("Statistics:")
print(f"Relevant + Factual + Helpful: {fmt(count_relevant_factual_helpful, valid_sample_count)}")
print(f"Relevant + Factual + Harmful: {fmt(count_relevant_factual_harmful, valid_sample_count)}")
print(f"Relevant + Factual + Neutral: {fmt(count_relevant_factual_neutral, valid_sample_count)}")
print(f"Broken samples: {broken_count}")

print("\nNeutral-case distribution:")
for k in ["✅✅", "❌✅", "✅❌", "❌❌"]:
    print(f"  {k}: {case_distribution_neutral[k]}")
# If you’d like to show unexpected case strings too:
for k in case_distribution_neutral:
    if k not in {"✅✅", "❌✅", "✅❌", "❌❌"}:
        print(f"  (other) {k!r}: {case_distribution_neutral[k]}")
