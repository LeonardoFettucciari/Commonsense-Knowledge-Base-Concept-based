def compute_metrics(
        ground_truths,
        answers_zeroshot = [],
        answers_zeroshot_with_knowledge = [],
        answers_fewshot = [],
        answers_fewshot_with_knowledge = [],
):
    correct_zeroshot = 0
    correct_zeroshot_with_knowledge = 0
    correct_fewshot = 0
    correct_fewshot_with_knowledge = 0

    for (
        gt,
        zs,
        zs_wk,
        fs,
        fs_wk,
    ) in zip(
        ground_truths,
        answers_zeroshot,
        answers_zeroshot_with_knowledge,
        answers_fewshot,
        answers_fewshot_with_knowledge,
    ):
        if gt == zs.strip():
            correct_zeroshot += 1
        if gt == zs_wk.strip():
            correct_zeroshot_with_knowledge += 1


        if gt == fs.strip():
            correct_fewshot += 1
        if gt == fs_wk.strip():
            correct_fewshot_with_knowledge += 1


    accuracy_zeroshot = correct_zeroshot/len(ground_truths)
    accuracy_zeroshot_with_knowledge = correct_zeroshot_with_knowledge/len(ground_truths)

    accuracy_fewshot = correct_fewshot/len(ground_truths)
    accuracy_fewshot_with_knowledge = correct_fewshot_with_knowledge/len(ground_truths)

    return {'accuracy_zeroshot': accuracy_zeroshot,
            'accuracy_zeroshot_with_knowledge': accuracy_zeroshot_with_knowledge,
            'accuracy_fewshot': accuracy_fewshot,
            'accuracy_fewshot_with_knowledge': accuracy_fewshot_with_knowledge,
            }