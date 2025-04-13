from datasets import Dataset
from src.utils.io_utils import load_local_file

anchors = []
positives = []
# Open a file, do preprocessing, filtering, cleaning, etc.
# and append to the lists

data = load_local_file()


dataset = Dataset.from_dict({
    "anchor": data[''],
    "positive": positives,
})