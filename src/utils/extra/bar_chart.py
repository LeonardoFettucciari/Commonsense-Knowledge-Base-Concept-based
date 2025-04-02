import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Patch

# Optional: set emoji-compatible font
# You may need to install one of these: 'Noto Emoji', 'Symbola', or 'Segoe UI Emoji'
rcParams['font.family'] = 'Symbola'  # Try 'Symbola' or 'Segoe UI Emoji' if needed

# Prompt types with emoji indicators
prompt_types = [
    'zeroshot',
    'zeroshot cot',
    'Zeroshot with knowledge 🧠🔍',
    'Zeroshot with knowledge 🧠🧹',
    'Zeroshot with knowledge 🧪🔍',
    'Zeroshot with knowledge 🧪🧹'
]

# Model names
models = [
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct'
]

# Accuracy data
accuracies = np.array([
    [68, 72, 70, 74],
    [71, 76, 73, 78],
    [75, 78, 80, 82],
    [78, 80, 83, 85],
    [69, 73, 72, 76],
    [82, 85, 87, 88]
])

# Config
num_groups = len(prompt_types)
num_bars = len(models)
bar_width = 0.2
group_width = bar_width * num_bars
group_gap = 0.3
group_starts = np.arange(num_groups) * (group_width + group_gap)

# Plot
fig, ax = plt.subplots(figsize=(15, 6))
for i in range(num_bars):
    bar_positions = group_starts + i * bar_width
    ax.bar(bar_positions, accuracies[:, i], width=bar_width, label=models[i])

# Labels and title
ax.set_xticks(group_starts + group_width / 2)
ax.set_xticklabels(prompt_types, rotation=30, ha='right')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy by Prompt Type')

# Legends
model_legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), title="Models")

emoji_handles = [
    Patch(color='none', label='🧠 Full KB'),
    Patch(color='none', label='🧪 Vera-filtered KB'),
    Patch(color='none', label='🔍 Retriever only'),
    Patch(color='none', label='🧹 CNER + Retriever')
]
emoji_legend = ax.legend(handles=emoji_handles, loc='upper left', bbox_to_anchor=(1.01, 0.55), title="Knowledge Configs")

ax.add_artist(model_legend)
ax.add_artist(emoji_legend)

# Clean look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# Export as SVG
plt.savefig("model_accuracy_by_prompt_type.svg", format='svg')
plt.show()
