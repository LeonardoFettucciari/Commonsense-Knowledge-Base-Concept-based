from pkg_resources import get_distribution
import re

input_file = "requirements.txt"
output_file = "requirements.txt"  # Overwrite the original file

pinned_lines = []

with open(input_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("git+"):
            pinned_lines.append(line)
            continue
        pkg_name = re.split(r'[<=>]', line)[0].strip()
        try:
            version = get_distribution(pkg_name).version
            pinned_lines.append(f"{pkg_name}=={version}")
        except Exception as e:
            pinned_lines.append(f"# {pkg_name} not installed: {e}")

with open(output_file, "w") as f:
    f.write("\n".join(pinned_lines) + "\n")

print(f"Pinned versions written to {output_file}")
