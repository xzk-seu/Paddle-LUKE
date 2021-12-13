import numpy as np
import paddle
import sys
from reprod_log import ReprodLogger
sys.path.append("../../../..")
from luke.modeling import LukeForEntityClassification as Model
# from luke.modeling import LukeModel as Model
np.random.seed(42)


if __name__ == "__main__":
    paddle.set_device("cpu")

    # def logger
    reprod_logger = ReprodLogger()

    model = Model.from_pretrained(
        "../../../../weight/pd/luke-large-finetuned-open-entity",
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

    criterion = paddle.nn.functional.binary_cross_entropy_with_logits
    fake_label = [0] * 9
    fake_label[0] = 1
    fake_label = paddle.to_tensor(fake_label, dtype=paddle.float32)
    fake_label = fake_label.reshape((1, 9))
    loss = criterion(out, fake_label)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")
