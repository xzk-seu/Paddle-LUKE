from tqdm import trange


def luke4oe_metric(
        batch_logits,
        batch_labels
):

    num_predicted = 0
    num_gold = 0
    num_correct = 0

    num_gold += batch_labels.sum()
    for logits, labels in zip(batch_logits, batch_labels):
        for logit, label in zip(logits, labels):
            if logit > 0:
                num_predicted += 1
                if label > 0:
                    num_correct += 1

    precision = num_correct / num_predicted
    recall = num_correct / num_gold
    f1 = 2 * precision * recall / (precision + recall)

    print(f"\n\nprecision: {precision} recall: {recall} f1: {f1}")
