def compute_metrics(
        ground_truths,
        answer_list = []
):
    correct_list = [0 for _ in range(len(answer_list))]

    for i, answers in enumerate(answer_list):
        for a in answers:
            if any(gt == a.strip() for gt in ground_truths):  # Check if any GT matches
                correct_list[i] += 1


    accuracy_list = []
    for c in correct_list:
        accuracy_list.append(c/len(ground_truths))

    return accuracy_list