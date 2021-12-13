import numpy as np
import torch
import torch.nn as nn
from reprod_log import ReprodLogger
from transformers.models.luke import LukeForEntityClassification as Model

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

    criterion = torch.nn.BCEWithLogitsLoss()
    fake_label = [0] * 9
    fake_label[0] = 1
    fake_label = torch.tensor(fake_label, dtype=torch.float32)
    fake_label = fake_label.reshape((1, 9))
    loss = criterion(out, fake_label)

    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")
