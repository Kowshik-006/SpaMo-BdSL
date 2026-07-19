import numpy as np

data = np.load("preprocess/Phoenix14T/train_info.npy", allow_pickle=True)

item = data.item()[0]

for key, value in item.items():
    print(f"{key} : {value}")