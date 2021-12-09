import numpy as np


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
