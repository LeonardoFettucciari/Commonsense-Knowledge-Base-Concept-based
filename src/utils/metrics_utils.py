def compute_metrics(
        ground_truths,
        answer_list = []
):
    correct_list = [[0] for _ in range(len(answer_list))]

    for i, answers in enumerate(answer_list):
        for gt, a in zip(ground_truths, answers):
            if gt == a.strip():
                correct_list[i] += 1

    accuracy_list = []
    for c in correct_list:
        accuracy_list.append(c/len(ground_truths))

    return accuracy_list