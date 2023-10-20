import torch
from torch.utils.data import Dataset
import pickle


class XorDataset(Dataset):
    def __init__(self, hp, mode):

        super().__init__()

        self.hp = hp

        if mode == 'train':
            dataset_path = hp.data.train_dataset_path
        else:
            dataset_path = hp.data.valid_dataset_path

        full_dataset_path = f"{hp.dir_setting.data_dir}/{dataset_path}"

        with open (full_dataset_path, "rb") as f:
            all_data = pickle.load(f)

        self.X = torch.from_numpy(all_data["X"]).to(torch.float32)
        self.y = torch.from_numpy(all_data["y"]).to(torch.float32)

    def __len__(self):

        return self.y.shape[0]

    def __getitem__(self, index):

        X = self.X[index, :]
        y = self.y[index]
        data = {"X": X, "y": y}

        return data
