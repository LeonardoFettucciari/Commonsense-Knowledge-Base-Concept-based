def compute_metrics(
        ground_truths,
        answers_zeroshot = [],
        answers_zeroshot_with_knowledge_k1 = [],
        answers_zeroshot_with_knowledge_k3 = [],
        answers_zeroshot_with_knowledge_k5 = [],
        answers_zeroshot_with_knowledge_k10 = [],
        answers_fewshot = [],
        answers_fewshot_with_knowledge_k1 = [],
        answers_fewshot_with_knowledge_k3 = [],
        answers_fewshot_with_knowledge_k5 = [],
        answers_fewshot_with_knowledge_k10 = [],
):
    
    correct_zeroshot = 0
    correct_zeroshot_with_knowledge_k1 = 0
    correct_zeroshot_with_knowledge_k3 = 0
    correct_zeroshot_with_knowledge_k5 = 0
    correct_zeroshot_with_knowledge_k10 = 0
    correct_fewshot = 0
    correct_fewshot_with_knowledge_k1 = 0
    correct_fewshot_with_knowledge_k3 = 0
    correct_fewshot_with_knowledge_k5 = 0
    correct_fewshot_with_knowledge_k10 = 0

    for (
        gt,
        zs,
        zs_wk_k1,
        zs_wk_k3,
        zs_wk_k5,
        zs_wk_k10,
        fs,
        fs_wk_k1,
        fs_wk_k3,
        fs_wk_k5,
        fs_wk_k10
    ) in zip(
        ground_truths,
        answers_zeroshot,
        answers_zeroshot_with_knowledge_k1,
        answers_zeroshot_with_knowledge_k3,
        answers_zeroshot_with_knowledge_k5,
        answers_zeroshot_with_knowledge_k10,
        answers_fewshot,
        answers_fewshot_with_knowledge_k1,
        answers_fewshot_with_knowledge_k3,
        answers_fewshot_with_knowledge_k5,
        answers_fewshot_with_knowledge_k10
    ):

        if gt == zs.strip():
            correct_zeroshot += 1
        if gt == zs_wk_k1.strip():
            correct_zeroshot_with_knowledge_k1 += 1
        if gt == zs_wk_k3.strip():
            correct_zeroshot_with_knowledge_k3 += 1
        if gt == zs_wk_k5.strip():
            correct_zeroshot_with_knowledge_k5 += 1
        if gt == zs_wk_k10.strip():
            correct_zeroshot_with_knowledge_k10 += 1


        if gt == fs.strip():
            correct_fewshot += 1
        if gt == fs_wk_k1.strip():
            correct_fewshot_with_knowledge_k1 += 1
        if gt == fs_wk_k3.strip():
            correct_fewshot_with_knowledge_k3 += 1
        if gt == fs_wk_k5.strip():
            correct_fewshot_with_knowledge_k5 += 1
        if gt == fs_wk_k10.strip():
            correct_fewshot_with_knowledge_k10 += 1


    accuracy_zeroshot = correct_zeroshot/len(ground_truths)
    accuracy_zeroshot_with_knowledge_k1 = correct_zeroshot_with_knowledge_k1/len(ground_truths)
    accuracy_zeroshot_with_knowledge_k3 = correct_zeroshot_with_knowledge_k3/len(ground_truths)
    accuracy_zeroshot_with_knowledge_k5 = correct_zeroshot_with_knowledge_k5/len(ground_truths)
    accuracy_zeroshot_with_knowledge_k10 = correct_zeroshot_with_knowledge_k10/len(ground_truths)

    accuracy_fewshot = correct_fewshot/len(ground_truths)
    accuracy_fewshot_with_knowledge_k1 = correct_fewshot_with_knowledge_k1/len(ground_truths)
    accuracy_fewshot_with_knowledge_k3 = correct_fewshot_with_knowledge_k3/len(ground_truths)
    accuracy_fewshot_with_knowledge_k5 = correct_fewshot_with_knowledge_k5/len(ground_truths)
    accuracy_fewshot_with_knowledge_k10 = correct_fewshot_with_knowledge_k10/len(ground_truths)

    return {'accuracy_zeroshot': accuracy_zeroshot,
            'accuracy_zeroshot_with_knowledge_k1': accuracy_zeroshot_with_knowledge_k1,
            'accuracy_zeroshot_with_knowledge_k3': accuracy_zeroshot_with_knowledge_k3,
            'accuracy_zeroshot_with_knowledge_k5': accuracy_zeroshot_with_knowledge_k5,
            'accuracy_zeroshot_with_knowledge_k10': accuracy_zeroshot_with_knowledge_k10,
            'accuracy_fewshot': accuracy_fewshot,
            'accuracy_fewshot_with_knowledge_k1': accuracy_fewshot_with_knowledge_k1,
            'accuracy_fewshot_with_knowledge_k3': accuracy_fewshot_with_knowledge_k3,
            'accuracy_fewshot_with_knowledge_k5': accuracy_fewshot_with_knowledge_k5,
            'accuracy_fewshot_with_knowledge_k10': accuracy_fewshot_with_knowledge_k10,
            }