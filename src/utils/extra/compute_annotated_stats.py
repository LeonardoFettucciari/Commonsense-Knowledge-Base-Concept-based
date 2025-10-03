import csv
from collections import Counter

csv_path = "outputs/annotations/stats/stats.csv"
broken_count = 0
count_relevant_factual_helpful = 0
count_relevant_factual_harmful = 0
count_relevant_factual_neutral = 0
valid_sample_count = 0
case_distribution_neutral = Counter({"✅✅": 0, "❌✅": 0, "✅❌": 0, "❌❌": 0})

# --------------------------------------------
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        valid_status = row.get("Valid question?", "")
        relevant     = row.get("Relevant?", "")
        factual      = row.get("Factual?", "")
        helpfulness  = row.get("Helpful?", "")

        if valid_status == "Broken":
            broken_count += 1
            continue
        valid_sample_count += 1

        if relevant == "Relevant" and factual == "Factual":
            if helpfulness == "Helpful":
                count_relevant_factual_helpful += 1
            elif helpfulness == "Harmful":
                count_relevant_factual_harmful += 1
            elif helpfulness == "Neutral":
                count_relevant_factual_neutral += 1

        if helpfulness == "Neutral":
            case_string = row.get("case", "")
            if case_string in case_distribution_neutral:
                case_distribution_neutral[case_string] += 1
            else:
                case_distribution_neutral[case_string] += 1

# Print results
def fmt(count, total):          
    return f"{count} ({(count/total*100):.2f}%)" if total else f"{count} (0.00%)"

print("Statistics:")
print(f"Relevant + Factual + Helpful: {fmt(count_relevant_factual_helpful, valid_sample_count)}")
print(f"Relevant + Factual + Harmful: {fmt(count_relevant_factual_harmful, valid_sample_count)}")
print(f"Relevant + Factual + Neutral: {fmt(count_relevant_factual_neutral, valid_sample_count)}")
print(f"Broken samples: {broken_count}")

print("\nNeutral-case distribution:")
for k in ["✅✅", "❌✅", "✅❌", "❌❌"]:
    print(f"  {k}: {case_distribution_neutral[k]}")

for k in case_distribution_neutral:
    if k not in {"✅✅", "❌✅", "✅❌", "❌❌"}:
        print(f"  (other) {k!r}: {case_distribution_neutral[k]}")
