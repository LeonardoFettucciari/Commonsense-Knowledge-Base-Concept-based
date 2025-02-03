import csv

def save_output_to_file(relative_path,
                        samples,
                        all_outputs,
                        ner_results,
                        unique_words_all_samples,
                        ):

    with open(relative_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            'id',
            'question',
            'choices',
            'ground_truth',
            'ner_results',
            'unique_words',
            'wordnet_synsets',
            'wordnet_definitions',
            'statements',
        ]
            
        tsv_writer = csv.writer(file, delimiter="\t")
        tsv_writer.writerow(fieldnames)

        for sample, outputs, ner, unique_words in zip(samples, all_outputs, ner_results, unique_words_all_samples):
            row_list = [
                sample['id'],
                sample['question'],
                "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                sample['answerKey'],
                "\n".join(f"{n['word']}@{n['entity_group']}" for n in ner),
                "\n".join(uw for uw in unique_words),
                "\n".join(o["synset"] for o in outputs),
                "\n".join(o["definition"] for o in outputs),
                "\n".join("\n".join(o["statements"]) for o in outputs),
            ]
            tsv_writer.writerow(row_list)