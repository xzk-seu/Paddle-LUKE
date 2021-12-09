from torch.utils.data.dataset import Dataset
import json
from transformers import LukeTokenizer
import numpy as np
# type2id = {"organization": 0, "event": 1, "time": 2, "place": 3, "object": 4,
#            "entity": 5, "group": 6, "person": 7, "location": 8}

temp = {
    "id2label": {
        "0": "entity",
        "1": "event",
        "2": "group",
        "3": "location",
        "4": "object",
        "5": "organization",
        "6": "person",
        "7": "place",
        "8": "time"
    }
}
type2id = {v: int(k) for k, v in temp["id2label"].items()}


class OpenEntityDataset(Dataset):
    def __init__(self, data_path, tokenizer=None, element_type=None):
        super().__init__()
        self.element_type = element_type
        if not tokenizer:
            self.tokenizer = LukeTokenizer.from_pretrained("../torch_model/luke-large-finetuned-open-entity")
        else:
            self.tokenizer = tokenizer

        def load_data_from_source(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

        self.data = load_data_from_source(data_path)

    def __getitem__(self, idx):
        example = self.data[idx]
        encoded_inputs = self.tokenizer(text=example["sent"], entity_spans=[(example["start"], example["end"])])
        label = [0.0] * 9
        for e in example["labels"]:
            idx = type2id[e]
            label[idx] = 1
        # label = np.array(label, dtype="float32")
        encoded_inputs["label"] = label
        return encoded_inputs

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d_path = "../dataset/OpenEntity/test.json"
    ds = OpenEntityDataset(d_path)  # paddle.io.Dataset
    # test_ds = MapDataset(ds)    # paddlenlp.datasets.MapDataset
