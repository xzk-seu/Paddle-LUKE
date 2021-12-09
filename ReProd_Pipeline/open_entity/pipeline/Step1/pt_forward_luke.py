import numpy as np
import torch
from reprod_log import ReprodLogger
from transformers.models.luke import LukeForEntityClassification as Model
# from transformers.models.luke import LukeModel as Model
np.random.seed(42)


if __name__ == "__main__":
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
    fake_data = {k: torch.tensor(fake_data[k]) for k in keys}
    # fake_data = torch.load(fake_data)

    # forward

    # model = model.luke
    outputs = model(**fake_data, return_dict=True)
    out = outputs.logits

    # out = model(**fake_data)[0]
    # out = model(**fake_data, return_dict=True)
    #
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_torch.npy")
