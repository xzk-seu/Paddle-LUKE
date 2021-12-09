import numpy as np
import paddle
import torch
from luke.modeling import LukeForEntityClassification as PDModel
from reprod_log import ReprodLogger
from transformers import AdamW
from transformers.models.luke import LukeForEntityClassification as PTModel
from tqdm import trange


def pd_train_some_iters(
        fake_data,
        fake_label,
        max_iter=2
):
    model = PDModel.from_pretrained(
        "../../../../pretrained_model/luke-large-finetuned-open-entity",
    )
    model.eval()
    criterion = paddle.nn.functional.binary_cross_entropy_with_logits
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=3e-5,
        parameters=model.parameters(),
        weight_decay=1e-2,
        epsilon=1e-6,
        apply_decay_param_fun=lambda x: x in decay_params,
    )
    loss_list = []
    for idx in trange(max_iter):
        fake_data = {k: paddle.to_tensor(fake_data[k]) for k in keys}
        labels = paddle.to_tensor(fake_label)

        outputs = model(**fake_data, return_dict=True)
        out = outputs["logits"]

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss)
    return loss_list


def hf_train_some_iters(fake_data, fake_label, max_iter=2):
    model = PTModel.from_pretrained(
        "../../../../torch_model/luke-large-finetuned-open-entity",
    )
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

    loss_list = []
    for idx in trange(max_iter):
        fake_data = {k: torch.tensor(fake_data[k]) for k in keys}
        labels = torch.from_numpy(fake_label)

        outputs = model(**fake_data, return_dict=True)
        out = outputs.logits

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss)
    return loss_list


if __name__ == "__main__":
    print("Start training")
    paddle.set_device("cpu")

    npzfile = np.load("../fake_data/fake_data.npz")
    keys = npzfile.files
    fake_data = {k: npzfile[k] for k in keys}

    fake_label = np.load("../fake_data/fake_label.npy")
    hf_reprod_logger = ReprodLogger()
    hf_loss_list = hf_train_some_iters(fake_data, fake_label, 10)
    for idx, loss in enumerate(hf_loss_list):
        hf_reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    hf_reprod_logger.save("bp_align_torch.npy")

    pd_reprod_logger = ReprodLogger()
    pd_loss_list = hf_train_some_iters(fake_data, fake_label, 10)
    for idx, loss in enumerate(pd_loss_list):
        pd_reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    pd_reprod_logger.save("bp_align_paddle.npy")
