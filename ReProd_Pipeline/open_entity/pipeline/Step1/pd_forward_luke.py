import numpy as np
import paddle
import sys

sys.path.append("../../../..")
from luke.modeling import LukeForEntityClassification as Model
# from luke.modeling import LukeModel as Model
np.random.seed(42)

from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("cpu")

    # def logger
    reprod_logger = ReprodLogger()

    model = Model.from_pretrained(
        "../../../../torch_model/luke-large-finetuned-open-entity",
    )
    model.eval()
    # read or gen fake dataset
    npzfile = np.load("../fake_data/fake_data.npz")
    keys = npzfile.files
    fake_data = {k: npzfile[k] for k in keys}
    fake_data = {k: paddle.to_tensor(fake_data[k]) for k in keys}

    # forward
    # model = model.luke
    outputs = model(**fake_data, return_dict=True)
    out = outputs["logits"]

    # out = model(**fake_data, return_dict=True)
    #
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_paddle.npy")
