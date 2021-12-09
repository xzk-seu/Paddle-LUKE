import numpy as np
import paddle
import torch
from paddle.io import DataLoader as PDDataLoader
from reprod_log import ReprodDiffHelper, ReprodLogger
from torch.utils.data.dataloader import DataLoader as PTDataLoader

from Dataset_utils.OpenEntityDataset import OpenEntityDataset as PDDataset
from Dataset_utils.OpenEntityDataset4pt import OpenEntityDataset as PTDataset

BATCH_SIZE = 32

# DATA_DIR = "./dataset/OpenEntity"
d_path = "dataset/OpenEntity/train.json"


# 重写collate_fn函数，其输入为一个batch的sample数据
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    entity_ids = [item['entity_ids'] for item in batch]
    entity_position_ids = [item['entity_position_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    entity_attention_mask = [item['entity_attention_mask'] for item in batch]
    # 因为token_list是一个变长的数据，所以需要用一个list来装这个batch的token_list
    # token_lists = [item['token_list'] for item in batch]

    max_seq_len = max([len(x) for x in input_ids])
    input_ids = [item + [1] * (max_seq_len - len(item)) for item in input_ids]
    attention_mask = [item + [0] * (max_seq_len - len(item)) for item in attention_mask]

    # 每个label是一个int，我们把这个batch中的label也全取出来，重新组装
    labels = [item['label'] for item in batch]
    # # 把labels转换成Tensor
    # labels = torch.Tensor(labels)
    res = {
        'input_ids': input_ids,
        'entity_ids': entity_ids,
        'entity_position_ids': entity_position_ids,
        'attention_mask': attention_mask,
        'entity_attention_mask': entity_attention_mask,
        'label': labels
    }
    res = {k: np.array(v) for k, v in res.items()}
    return res


def build_paddle_data_pipeline():
    dataset = PDDataset(d_path)  # paddle.io.Dataset

    loader = PDDataLoader(dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          # drop_last=True,
                          collate_fn=collate_fn,
                          num_workers=0)

    dataset_test, data_loader_test = dataset, loader

    return dataset_test, data_loader_test


def build_torch_data_pipeline():
    dataset_test = PTDataset(d_path)

    train_loader = PTDataLoader(dataset=dataset_test,  # 传递数据集
                                batch_size=BATCH_SIZE,  # 一个小批量容量是多少
                                shuffle=False,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                                collate_fn=collate_fn,
                                num_workers=0)  # 需要几个进程来一次性读取这个小批量数据

    dataset_test, data_loader_test = dataset_test, train_loader
    return dataset_test, data_loader_test


def test_data_pipeline():
    keys = list()
    diff_helper = ReprodDiffHelper()
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline()
    torch_dataset, torch_dataloader = build_torch_data_pipeline()

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    # random choose 5 images and check
    for idx in range(5):
        rnd_idx = np.random.randint(0, len(paddle_dataset))
        example = paddle_dataset[rnd_idx]
        keys = example.keys()
        for k in keys:
            paddle_data = example[k]
            paddle_data = paddle.to_tensor(paddle_data)
            paddle_data = paddle_data.numpy()
            torch_data = torch_dataset[rnd_idx]
            torch_data = torch_data[k]
            torch_data = torch.tensor(torch_data)
            torch_data = torch_data.detach().numpy()

            logger_paddle_data.add(f"dataset_{idx}_{k}",
                                   paddle_data)

            logger_torch_data.add(f"dataset_{idx}_{k}",
                                  torch_data)

    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx >= 5:
            break
        for i, k in enumerate(keys):
            logger_paddle_data.add(f"dataloader_{idx}_{k}",
                                   paddle_batch[k].numpy())
            logger_torch_data.add(f"dataloader_{idx}_{k}",
                                  torch_batch[k])

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report()


if __name__ == "__main__":
    test_data_pipeline()
