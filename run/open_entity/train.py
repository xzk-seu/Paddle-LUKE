import datetime
import os
import random
import time

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.optimizer import AdamW

import utils
import sys

sys.path.append("../..")

from Dataset_utils.OpenEntityDataset import OpenEntityDataset
from Dataset_utils.collate_fn import collate_fn
from luke.modeling import LukeForEntityClassification
from luke.tokenizer import LukeTokenizer
from paddlenlp.metrics.glue import AccuracyAndF1


def train_one_epoch(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        data_loader,
        epoch,
        print_freq,
        scaler=None, ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(
            window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "sentence/s", utils.SmoothedValue(
            window_size=10, fmt="{value}"))

    header = "Epoch: [{}]".format(epoch)
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = {k: paddle.to_tensor(v) for k, v in batch.items()}
        labels = batch.pop("label")
        start_time = time.time()
        with paddle.amp.auto_cast(
                enable=scaler is not None,
                custom_white_list=["layer_norm", "softmax", "gelu"], ):
            outputs = model(**batch, return_dict=True)
            logits = outputs["logits"]
            labels = paddle.cast(labels, dtype=paddle.float32)
            loss = criterion(logits, labels)

        optimizer.clear_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        batch_size = batch["input_ids"].shape[0]
        metric_logger.update(loss=loss.item(), lr=lr_scheduler.get_lr())
        metric_logger.meters["sentence/s"].update(batch_size /
                                                  (time.time() - start_time))


def evaluate(model, criterion, data_loader, metric, print_freq=100):
    model.eval()
    metric.reset()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with paddle.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            batch = {k: paddle.to_tensor(v) for k, v in batch.items()}
            labels = batch.pop("label")

            outputs = model(**batch, return_dict=True)
            logits = outputs["logits"]
            labels = paddle.cast(labels, dtype=paddle.float32)
            loss = criterion(logits, labels)

            metric_logger.update(loss=loss.item())

            logits = logits.reshape((-1,))
            labels = labels.reshape((-1,))
            positive = (logits > 0).astype(paddle.int64)
            correct = (positive == labels).astype(np.float32)

            metric.label = labels
            metric.preds_pos = positive
            metric.update(correct)

        # acc_global_avg = metric.accumulate()
        pd_accuracy, pd_precision, pd_recall, pd_f1, _ = metric.accumulate()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(" * f1 {acc_global_avg:.6f}".format(
        acc_global_avg=pd_f1))
    return pd_f1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def load_data(args, tokenizer):
    print("Loading dataset")
    d_path = args.data_set_dir
    train_set_path = os.path.join(d_path, args.train_file)
    train_ds = OpenEntityDataset(train_set_path, tokenizer)

    train_loader = DataLoader(dataset=train_ds,  # 传递数据集
                              batch_size=args.batch_size,  # 一个小批量容量是多少
                              shuffle=False,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                              collate_fn=collate_fn,
                              num_workers=args.workers)  # 需要几个进程来一次性读取这个小批量数据

    val_set_path = os.path.join(d_path, args.test_file)
    validation_ds = OpenEntityDataset(val_set_path, tokenizer)

    validation_loader = DataLoader(dataset=validation_ds,  # 传递数据集
                                   batch_size=args.batch_size,  # 一个小批量容量是多少
                                   shuffle=False,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                                   collate_fn=collate_fn,
                                   num_workers=args.workers)  # 需要几个进程来一次性读取这个小批量数据

    return train_ds, validation_ds, train_loader, validation_loader


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    print(args)
    scaler = None
    if args.fp16:
        scaler = paddle.amp.GradScaler()
    paddle.set_device(args.device)

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = LukeTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset, validation_dataset, train_data_loader, validation_data_loader = load_data(args, tokenizer)

    print("Creating model")
    model = LukeForEntityClassification.from_pretrained(args.model_name_or_path)
    print("Creating criterion")
    criterion = paddle.nn.functional.binary_cross_entropy_with_logits

    print("Creating lr_scheduler")
    lr_scheduler = utils.get_scheduler(
        learning_rate=args.lr,
        scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_data_loader), )

    print("Creating optimizer")
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in no_decay)
    ]
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        epsilon=1e-6,
        apply_decay_param_fun=lambda x: x in decay_params, )
    metric = AccuracyAndF1()
    metric.reset()

    if args.test_only:
        evaluate(model, criterion, validation_data_loader, metric)
        return

    print("Start training")
    start_time = time.time()
    best_accuracy = 0.0
    for epoch in range(args.num_train_epochs):

        train_one_epoch(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            train_data_loader,
            epoch,
            args.print_freq,
            scaler, )
        acc = evaluate(model, criterion, validation_data_loader, metric)
        best_accuracy = max(best_accuracy, acc)
        if args.output_dir:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return best_accuracy


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Paddle Entity Classification Training", add_help=add_help)
    parser.add_argument(
        "--task_name",
        default="open entity",
        help="the name of the glue task to train on.")
    parser.add_argument(
        "--model_name_or_path",
        default="./pretrained_model/luke-large-finetuned-open-entity",
        help="path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--device", default="gpu", help="device")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        ), )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="number of total epochs to run")
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        help="number of dataset loading workers (default: 16)", )
    parser.add_argument(
        "--lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="weight decay (default: 1e-2)",
        dest="weight_decay", )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="the scheduler type to use.",
        choices=["linear", "cosine", "polynomial"], )
    parser.add_argument(
        "--num_warmup_steps",
        default=0,
        type=int,
        help="number of steps for the warmup in the lr scheduler.", )
    parser.add_argument(
        "--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--output_dir", default="outputs", help="path where to save")
    parser.add_argument(
        "--test_only",
        help="only test the model",
        action="store_true", )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="a seed for reproducible training.")
    # Mixed precision training parameters
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="whether or not mixed precision training")
    parser.add_argument(
        "--data_set_dir",
        default="./dataset/OpenEntity"
    )
    parser.add_argument(
        "--train_file",
        default="train.json"
    )
    parser.add_argument(
        "--test_file",
        default="test.json"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    acc = main(args)
