import numpy as np
import paddle
import sys
sys.path.append("../../../..")
from luke.modeling import LukeModel

from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("cpu")

    # def logger
    reprod_logger = ReprodLogger()

    model = LukeModel.from_pretrained(
        "../../../../torch_model/luke-base",
        # num_classes=2
    )
    # classifier_weights = paddle.load(
    #     "../classifier_weights/paddle_classifier_weights.bin")
    # model.load_dict(classifier_weights)
    model.eval()
    # read or gen fake dataset
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)
    # forward
    out = model(fake_data)[0]
    #
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_paddle.npy")
