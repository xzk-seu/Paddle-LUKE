import numpy as np

np.random.seed(42)


def gen_fake_data():
    """
    random.randint(low, high=None, size=None, dtype=int)
    :return:
    :rtype:
    """
    fake_data = np.random.randint(1, 50267, size=(4, 64)).astype(np.int64)
    fake_label = np.array([0, 1, 1, 0]).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)


if __name__ == "__main__":
    gen_fake_data()
