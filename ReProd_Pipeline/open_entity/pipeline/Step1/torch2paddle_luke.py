from collections import OrderedDict

import numpy as np
import paddle
import torch
from transformers import LukeModel as PTLukeModel
import sys
sys.path.append("../../../..")
from luke.modeling import LukeModel as PDLukeModel


def convert_pytorch_checkpoint_to_paddle(
        pytorch_checkpoint_path="pytorch_model.bin",
        paddle_dump_path="model_state.pdparams"
):

    pytorch_state_dict = torch.load(
        pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = dict()

    filter_wight_list = ["dense.", ".value.", ".query.", ".key."]
    for k, v in pytorch_state_dict.items():
        is_transpose = False

        if "weight" in k and any([y in k for y in filter_wight_list]):
            v = v.transpose(0, 1)
            is_transpose = True

        old_k = k
        # if "luke." not in k and "cls." not in k and "classifier" not in k:
        #     k = "luke." + k

        print(f"Converting: {old_k} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)


def compare(out_torch, out_paddle):
    out_torch = out_torch.detach().numpy()
    out_paddle = out_paddle.detach().numpy()
    assert out_torch.shape == out_paddle.shape
    abs_dif = np.abs(out_torch - out_paddle)
    mean_dif = np.mean(abs_dif)
    max_dif = np.max(abs_dif)
    min_dif = np.min(abs_dif)
    print("mean_dif:{}".format(mean_dif))
    print("max_dif:{}".format(max_dif))
    print("min_dif:{}".format(min_dif))


def test_forward():
    paddle.set_device("cpu")
    model_torch = PTLukeModel.from_pretrained("../../../../torch_model/luke-base")
    model_paddle = PDLukeModel.from_pretrained("../../../../torch_model/luke-base")
    model_torch.eval()
    model_paddle.eval()
    np.random.seed(42)
    x = np.random.randint(
        1, model_paddle.config["vocab_size"], size=(4, 64))
    input_torch = torch.tensor(x, dtype=torch.int64)
    out_torch = model_torch(input_torch)[0]

    input_paddle = paddle.to_tensor(x, dtype=paddle.int64)
    out_paddle = model_paddle(input_paddle)[0]

    print("torch result shape:{}".format(out_torch.shape))
    print("paddle result shape:{}".format(out_paddle.shape))
    compare(out_torch, out_paddle)


if __name__ == "__main__":
    # convert_pytorch_checkpoint_to_paddle(
    #     "../../../luke-base/pytorch_model.bin",
    #     "../../../luke-base/model_state.pdparams")
    test_forward()
    # torch result shape:torch.Size([4, 64, 30522])
    # paddle result shape:[4, 64, 30522]
    # mean_dif:1.666686512180604e-05
    # max_dif:0.00015211105346679688
    # min_dif:0.0
