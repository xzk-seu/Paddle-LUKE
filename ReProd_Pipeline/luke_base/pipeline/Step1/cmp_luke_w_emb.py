from collections import OrderedDict

import numpy as np
import paddle
import torch
from transformers import LukeModel as PTLukeModel
from transformers.models.luke.modeling_luke import LukeEmbeddings as PTLukeEmbeddings
from transformers.models.luke.configuration_luke import LukeConfig
import sys
sys.path.append("../../../..")
from luke.modeling import LukeModel as PDLukeModel
from luke.modeling import LukeEmbeddings as PDLukeEmbeddings
from paddle.nn.layer import Embedding as PDEmbedding

from torch import nn as ptnn
from paddle import nn as pdnn

from torch.nn.parameter import Parameter as PTParameter


np.random.seed(42)


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

    input_var = {
        "input_ids": np.random.randint(1, 50267, size=(4, 64)),
        # "position_ids": np.random.randint(1, 63, size=(4, 64)),
        "token_type_ids": np.random.randint(0, 1, size=(4, 64)),
        # "input_ids": np.random.randint(1, 50267, size=(4, 64))
    }
    input_var_torch = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_var.items()}
    input_var_paddle = {k: paddle.to_tensor(v, dtype=paddle.int64) for k, v in input_var.items()}

    model_torch = PTLukeModel.from_pretrained("../../../../torch_model/luke-base")
    model_paddle = PDLukeModel.from_pretrained("../../../../torch_model/luke-base")

    luke_emb_torch = model_torch.embeddings
    luke_emb_paddle = model_paddle.embeddings

    luke_emb_torch.eval()
    luke_emb_paddle.eval()

    # x = np.random.randint(
    #     1, 50267, size=(4, 64))
    # input_torch = torch.tensor(x, dtype=torch.int64)
    out_torch = luke_emb_torch(input_ids=input_var_torch['input_ids'],
                               token_type_ids=input_var_torch['token_type_ids']
                               )

    # input_paddle = paddle.to_tensor(x, dtype=paddle.int64)
    out_paddle = luke_emb_paddle(
        input_ids=input_var_paddle['input_ids'],
        token_type_ids=input_var_paddle['token_type_ids']
    )  # [0]

    print("torch result shape:{}".format(out_torch.shape))
    print("paddle result shape:{}".format(out_paddle.shape))
    compare(out_torch, out_paddle)


def emb_test():
    vocab_size = 50000
    hidden_size = 768

    weight = np.random.normal(0, 0.02, (vocab_size, hidden_size)).astype("float32")

    paddle.set_device("cpu")
    x = np.random.randint(1, 128, size=(4, 64))
    pt_emb = ptnn.Embedding(50000, 768,
                            # _weight=
                            )
    pd_emb = pdnn.Embedding(50000, 768,
                            # weight_attr=
                            )

    pt_emb.weight = PTParameter(torch.tensor(weight))
    pd_emb.weight.set_value(paddle.to_tensor(weight))

    pt_emb.eval()
    pd_emb.eval()

    in_torch = torch.tensor(x, dtype=torch.int64)
    out_torch = pt_emb(in_torch)

    in_paddle = paddle.to_tensor(x, dtype=paddle.int64)
    out_paddle = pd_emb(in_paddle)

    print("torch result shape:{}".format(out_torch.shape))
    print("paddle result shape:{}".format(out_paddle.shape))
    compare(out_torch, out_paddle)


def drop_test():
    from torch.nn import Dropout as PTDropout
    from paddle.nn import Dropout as PDDropout

    # weight = np.random.normal(0, 0.02, (vocab_size, hidden_size)).astype("float32")

    paddle.set_device("cpu")
    x = np.random.randint(1, 128, size=(4, 64))
    pt_drop = PTDropout()
    pd_drop = PDDropout()

    pd_drop.eval()
    pt_drop.eval()

    in_torch = torch.tensor(x, dtype=torch.float32)
    out_torch = pt_drop(in_torch)

    in_paddle = paddle.to_tensor(x, dtype=paddle.float32)
    out_paddle = pd_drop(in_paddle)

    print("torch result shape:{}".format(out_torch.shape))
    print("paddle result shape:{}".format(out_paddle.shape))
    compare(out_torch, out_paddle)


if __name__ == "__main__":
    # emb_test()  # 对两个框架的embedding进行对比
    test_forward()  # 对模型的embedding进行对比
    # drop_test() # 对两个框架的dropout进行对比
    # torch result shape:torch.Size([4, 64, 30522])
    # paddle result shape:[4, 64, 30522]
    # mean_dif:1.666686512180604e-05
    # max_dif:0.00015211105346679688
    # min_dif:0.0
