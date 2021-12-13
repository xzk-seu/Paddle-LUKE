#%%
import numpy as np
from datasets import load_metric
logits = np.random.normal(0, 1, size=(32, 9)).astype("float32")
labels = np.random.randint(0, 2, size=(32, 9)).astype("int64")

metric = load_metric("f1_luke4oe.py")

# predictions=torch.from_numpy(logits).argmax(dim=-1)
predictions = (logits > 0).astype(np.int32)
predictions = predictions.reshape((32*9))
labels = labels.reshape((32*9))

import torch
metric.add_batch(predictions=torch.from_numpy(predictions),
                 references=torch.from_numpy(labels)
                 )


f1 = metric.compute()

