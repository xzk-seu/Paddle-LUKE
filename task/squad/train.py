import datetime
import logging
import os
import random
import time
import joblib

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.optimizer import AdamW
from paddlenlp.metrics import squad

from Dataset_utils.SquadDataset import SquadDataset
from luke.tokenizer import LukeTokenizer
from task.squad.utils.dataset import SquadV1Processor
from task.squad.utils.feature import convert_examples_to_features
from task.squad.utils.utils import MetricLogger, SmoothedValue
from task.utils.entity_vocab import EntityVocab
from task.squad.utils.wiki_link_db import WikiLinkDB

logger = logging.getLogger(__name__)


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
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", SmoothedValue(
            window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "sentence/s", SmoothedValue(
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
    metric_logger = MetricLogger(delimiter="  ")
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
        squad.squad_evaluate([], [])
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


def load_examples(args, tokenizer, is_evaluate=False):
    # if args.local_rank not in (-1, 0) and not evaluate:
    #     torch.distributed.barrier()

    # if args.with_negative:
    #     processor = SquadV2Processor()
    # else:
    #     processor = SquadV1Processor()
    processor = SquadV1Processor()

    if is_evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    # bert_model_name = args.model_config.bert_model_name

    # segment_b_id = 1
    # add_extra_sep_token = False
    # if "roberta" in bert_model_name:
    #     segment_b_id = 0
    #     add_extra_sep_token = True
    segment_b_id = 0
    add_extra_sep_token = True

    args.entity_vocab = EntityVocab(args.entity_vocab_tsv)

    args.wiki_link_db = WikiLinkDB(args.wiki_link_db_file)
    args.model_redirect_mappings = joblib.load(args.model_redirects_file)
    args.link_redirect_mappings = joblib.load(args.link_redirects_file)

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        entity_vocab=args.entity_vocab,
        wiki_link_db=args.wiki_link_db,
        model_redirect_mappings=args.model_redirect_mappings,
        link_redirect_mappings=args.link_redirect_mappings,
        max_seq_length=args.max_seq_length,
        max_mention_length=args.max_mention_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        min_mention_link_prob=args.min_mention_link_prob,
        segment_b_id=segment_b_id,
        add_extra_sep_token=add_extra_sep_token,
        is_training=not is_evaluate,
    )

    # if args.local_rank == 0 and not is_evaluate:
    #     torch.distributed.barrier()

    def collate_fn(batch):
        def pad_sequence(x, padding_value):
            max_seq_len = max([len(i) for i in x])
            res = [item + [padding_value] * (max_seq_len - len(item)) for item in x]
            return res

        def create_padded_sequence(attr_name, padding_value):
            x = [getattr(o[1], attr_name) for o in batch]
            res = pad_sequence(x, padding_value)
            res = paddle.to_tensor(res)
            return res

        ret = dict(
            word_ids=create_padded_sequence("word_ids", tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence("entity_ids", 0)[:, : args.max_entity_length],
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0)[:, : args.max_entity_length],
            entity_position_ids=create_padded_sequence("entity_position_ids", -1)[:, : args.max_entity_length, :],
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0)[:, : args.max_entity_length],
        )
        # if args.no_entity:
        #     ret["entity_attention_mask"].fill_(0)

        if is_evaluate:
            ret["example_indices"] = paddle.to_tensor([o[0] for o in batch], dtype=paddle.int64)
        else:
            ret["start_positions"] = paddle.to_tensor([o[1].start_positions[0] for o in batch], dtype=paddle.int64)
            ret["end_positions"] = paddle.to_tensor([o[1].end_positions[0] for o in batch], dtype=paddle.int64)

        return ret
    dataset = SquadDataset(features)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, processor


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Paddle Squad Training", add_help=add_help)
    parser.add_argument(
        "--task_name",
        default="squad",
        help="the name of the task to train on.")
    parser.add_argument(
        "--model_name_or_path",
        default="./pd/luke-large-finetuned-open-entity",
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
        help="number of total epochs to task")
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

    parser.add_argument("--data_dir", default="./dataset/squad")
    parser.add_argument("--entity_vocab_tsv", default="weight/pd/luke-for-squad/entity_vocab.tsv")
    parser.add_argument("--max_seq_length", default=512)
    parser.add_argument("--max_mention_length", default=30)
    parser.add_argument("--doc_stride", default=128)
    parser.add_argument("--max_query_length", default=64)
    parser.add_argument("--min_mention_link_prob", default=0.01)
    parser.add_argument("--max_entity_length", default=128)

    parser.add_argument("--link-redirects-file", default="enwiki_20160305_redirects.pkl")
    parser.add_argument("--model-redirects-file", default="enwiki_20181220_redirects.pkl")
    parser.add_argument("--wiki-link-db-file", default="enwiki_20160305.pkl")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    acc = main(args)
