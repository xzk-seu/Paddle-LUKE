from paddle.io import Dataset


class SquadDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = list(enumerate(data))

    def __getitem__(self, idx):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d_path = "../dataset/OpenEntity/test.json"
    ds = SquadDataset(d_path)
