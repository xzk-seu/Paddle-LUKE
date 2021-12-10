from paddle.io import Dataset
from paddlenlp.datasets import MapDataset
import json
import numpy as np
from luke.tokenizer import LukeTokenizer
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
    def __init__(self, data_path, tokenizer=None):
        super().__init__()

        def load_data_from_source(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

        self.data = load_data_from_source(data_path)
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = LukeTokenizer.from_pretrained(
                "../weight/pd/luke-large-finetuned-open-entity")

    def __getitem__(self, idx):
        example = self.data[idx]
        encoded_inputs = self.tokenizer(text=example["sent"], entity_spans=[(example["start"], example["end"])],
                                        max_seq_len=128)
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
    test_ds = MapDataset(ds)    # paddlenlp.datasets.MapDataset
