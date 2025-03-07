import csv
import os

input_path = os.path.dirname(os.path.abspath(__file__))
output_path = f"{os.path.dirname(os.path.abspath(__file__))}/output"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

for filename in os.listdir(input_path):
    if filename.endswith(".tsv"):
        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, filename)
        
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8", newline="") as outfile:
            
            reader = csv.DictReader(infile, delimiter='\t')
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
            
            writer.writeheader()
            
            for row in reader:
                prompt = row.get("prompt", "")
                if '\n\n' in prompt:
                    # Remove everything before and including the first newline
                    row["prompt"] = prompt.split('\n\n', 1)[1]
                else:
                    # If there's no newline, the prompt stays as it is or becomes empty
                    row["prompt"] = ""
                writer.writerow(row)

        print(f"Processed: {filename}")

print("✅ All files processed.")
