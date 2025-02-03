import os
import csv

def save_output_to_file(relative_path,
                        samples,
                        cleaned_statements,
                        ner_results,
                        synsets_all_samples,
                        definitions_all_samples,
                        unique_words_all_samples,
                        ):

    with open(relative_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ['id',
                    'question',
                    'choices',

                    'statements',
                    'ner_results',
                    'unique_words',
                    'wordnet_synsets',
                    'wordnet_definitions',

                    'gold_truth',]
            
        tsv_writer = csv.writer(file, delimiter="\t")
        tsv_writer.writerow(fieldnames)

        for sample, statements, ner, synsets_list, definitions_list, unique_words in zip(
            samples,
            cleaned_statements,
            ner_results,
            synsets_all_samples,
            definitions_all_samples,
            unique_words_all_samples,
            ):

            row_list = [sample['id'],
                        sample['question'],
                        "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),

                        "\n".join(s for s in statements),
                        "\n".join(f"{n['word']}@{n['entity_group']}" for n in ner),
                        "\n".join(uw for uw in unique_words),
                        "\n".join(s.name() for synsets in synsets_list for s in synsets),
                        "\n".join(d for definitions in definitions_list for d in definitions),
                        
                        sample['answerKey'],]
            tsv_writer.writerow(row_list)