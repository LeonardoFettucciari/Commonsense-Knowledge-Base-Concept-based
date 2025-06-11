import csv

# Load the CSV file
csv_path = "outputs/annotations/stats/stats.csv"  # replace with your actual file path

# Initialize counters
broken_count = 0
count_relevant_factual_helpful = 0
count_relevant_factual_harmful = 0
count_relevant_factual_neutral = 0
valid_sample_count = 0

# Read the CSV file
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        valid_status = row.get("Valid question?", "")
        relevant = row.get("Relevant?", "")
        factual = row.get("Factual?", "")
        helpfulness = row.get("Helpful?", "")

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

# Print results
print("Statistics:")

def format_count_and_percentage(count, total):
    if total == 0:
        return f"{count} (0.00%)"
    percentage = (count / total) * 100
    return f"{count} ({percentage:.2f}%)"

print(f"Relevant + Factual + Helpful: {format_count_and_percentage(count_relevant_factual_helpful, valid_sample_count)}")
print(f"Relevant + Factual + Harmful: {format_count_and_percentage(count_relevant_factual_harmful, valid_sample_count)}")
print(f"Relevant + Factual + Neutral: {format_count_and_percentage(count_relevant_factual_neutral, valid_sample_count)}")
print(f"Broken samples: {broken_count}")
