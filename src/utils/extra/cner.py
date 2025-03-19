from src.utils.model_utils import get_ner_pipeline
from src.utils.data_utils import concatenate_question_choices, extract_unique_words, from_words_to_synsets, synsets_from_samples
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

samples = ["An electric car runs on electricity via A. gasoline B. a power station C. electrical conductors D. fuel"]
ner_pipeline = get_ner_pipeline("Babelscape/cner-base")
ner_results = ner_pipeline(samples)

logging.info(ner_results)

# Extract unique words for each sample separately
unique_words_per_sample = [extract_unique_words(ner_result) for ner_result in ner_results]

logging.info(unique_words_per_sample)
# Convert words to synsets for each sample separately
synsets_per_sample = [from_words_to_synsets(unique_words) for unique_words in unique_words_per_sample]

logging.info(synsets_per_sample)

synsets_from_samples_per_sample = synsets_from_samples(samples)
logging.info(synsets_from_samples_per_sample)
