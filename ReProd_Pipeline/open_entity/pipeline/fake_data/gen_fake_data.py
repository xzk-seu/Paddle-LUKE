import numpy as np
import paddle
from transformers import LukeTokenizer
import torch
np.random.seed(42)


def get_torch_input():
    tokenizer = LukeTokenizer.from_pretrained("../../../../torch_model/luke-large-finetuned-open-entity")
    text = "Beyoncé lives in Los Angeles."
    entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
    x = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
    torch_input = dict(x)

    return torch_input


def get_paddle_input(torch_input_list):
    paddle_input_list = [paddle.to_tensor(t.detach().numpy(), dtype=paddle.int64)
                        for t in torch_input_list]
    return paddle_input_list


def get_torch_list(torch_input):
    token_type_ids = np.full((1, 11), 0, dtype=np.int64)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64)

    position_ids = np.arange(11, dtype=np.int64)
    position_ids = position_ids.reshape(1, 11)
    position_ids = torch.tensor(position_ids, dtype=torch.int64)

    entity_token_type_ids = np.full((1, 1), 0)
    entity_token_type_ids = torch.tensor(entity_token_type_ids, dtype=torch.int64)

    torch_input_list = [
        torch_input['input_ids'],
        torch_input['attention_mask'],
        token_type_ids,
        position_ids,
        torch_input['entity_ids'],
        torch_input['entity_attention_mask'],
        entity_token_type_ids,
        torch_input['entity_position_ids'],
    ]
    return torch_input_list


def get_torch_dict(torch_input):
    token_type_ids = np.full((1, 11), 0, dtype=np.int64)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64)

    position_ids = np.arange(11, dtype=np.int64)
    position_ids = position_ids.reshape(1, 11)
    position_ids = torch.tensor(position_ids, dtype=torch.int64)

    entity_token_type_ids = np.full((1, 1), 0)
    entity_token_type_ids = torch.tensor(entity_token_type_ids, dtype=torch.int64)
    temp_dict = dict(
        input_ids=torch_input['input_ids'],
        attention_mask=torch_input['attention_mask'],
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        entity_ids=torch_input['entity_ids'],
        entity_attention_mask=torch_input['entity_attention_mask'],
        entity_token_type_ids=entity_token_type_ids,
        entity_position_ids=torch_input['entity_position_ids'],
        # head_mask=None,
        # inputs_embeds=None,
        # labels=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
    )
    return temp_dict


def gen_fake_data():
    """
    random.randint(low, high=None, size=None, dtype=int)
    :return:
    :rtype:
    """
    torch_input = get_torch_input()
    torch_input_dict = get_torch_dict(torch_input)
    fake_data = {k: v.detach().numpy() for k, v in torch_input_dict.items()}
    # fake_data = [t.detach().numpy() for t in torch_input_list]

    np.savez("fake_data.npz", **fake_data)
    # np.save("fake_label.npy", fake_label)


def gen_fake_label():
    fake_label = [0] * 9
    fake_label[0] = 1
    fake_label = np.array(fake_label, dtype=np.float32)
    fake_label = fake_label.reshape((1, 9))
    np.save("fake_label.npy", fake_label)


if __name__ == "__main__":
    gen_fake_data()
    gen_fake_label()
