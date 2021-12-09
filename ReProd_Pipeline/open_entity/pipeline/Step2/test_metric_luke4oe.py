import numpy as np
import paddle
import torch
from datasets import load_metric
from paddle.metric import Accuracy, Precision, Recall
from reprod_log import ReprodLogger
from paddlenlp.metrics.glue import AccuracyAndF1
# from transformers


def generate():
    pd_metric = AccuracyAndF1()
    pd_metric.reset()
    hf_metric = load_metric("glue", "qqp")
    for i in range(4):
        logits = np.random.normal(0, 1, size=(32, 9)).astype(np.float32)
        logits = logits.reshape((32 * 9))
        positive = (logits > 0).astype(np.int64)
        labels = np.random.randint(0, 2, size=(32, 9)).astype(np.int64)
        labels = labels.reshape((32 * 9))
        correct = (positive == labels).astype(np.float32)
        # paddle metric

        # correct = pd_metric.compute(paddle.to_tensor(logits), paddle.to_tensor(labels))
        pd_metric.label = labels
        pd_metric.preds_pos = positive
        pd_metric.update(correct)
        # correct = pd_metric.compute(
        #     paddle.to_tensor(logits), paddle.to_tensor(labels))
        # pd_metric.update(correct)

        # hf metric
        hf_metric.add_batch(predictions=positive, references=labels)
        # hf_metric.add_batch(
        #     predictions=torch.from_numpy(logits).argmax(dim=-1),
        #     references=torch.from_numpy(labels), )

    pd_accuracy, pd_precision, pd_recall, pd_f1, _ = pd_metric.accumulate()
    hf_res = hf_metric.compute()
    hf_accuracy = hf_res["accuracy"]
    hf_f1 = hf_res["f1"]
    reprod_logger = ReprodLogger()
    reprod_logger.add("f1", np.array([pd_f1]))
    reprod_logger.save("metric_paddle.npy")
    reprod_logger = ReprodLogger()
    reprod_logger.add("f1", np.array([hf_f1]))
    reprod_logger.save("metric_torch.npy")


if __name__ == "__main__":
    generate()
