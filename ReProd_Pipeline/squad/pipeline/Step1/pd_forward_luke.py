import numpy as np
import paddle
import sys

from luke.modeling import LukeForReadingComprehension as Model
# from luke.modeling import LukeModel as Model
np.random.seed(42)

from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("cpu")

    # def logger
    reprod_logger = ReprodLogger()

    model = Model.from_pretrained(
        "weight/pd/luke-for-squad",
    )
    model.eval()
    # read or gen fake dataset
    npzfile = np.load("ReProd_Pipeline/squad/pipeline/fake_data/fake_data.npz")
    keys = npzfile.files
    fake_data = {k: npzfile[k] for k in keys}
    fake_data = {k: paddle.to_tensor(fake_data[k]) for k in keys}
    fake_data.pop("example_indices")

    # forward
    # model = model.luke
    outputs = model(**fake_data, return_dict=True)
    out = outputs[0]

    # out = model(**fake_data, return_dict=True)
    #
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_paddle.npy")
