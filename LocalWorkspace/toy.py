import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

def set_config(hyper_config, trainer_config):
    trainer_config["max_epochs"] = 30
    trainer_config["log_every_n_steps"] = 1
    hyper_config["lr"] = 1E-1
    hyper_config["log_freq"] = 1
    ansatz_config = [
        (4, 10, 0), 
        (4, 20, 0), 
        (4, 10, 1), 
        (4, 20, 1),
    ]
    return hyper_config, trainer_config, ansatz_config

class ToyDataset(Dataset):
    def __init__(self, x, y):
        from sklearn.datasets import load_iris
        self.x = x
        self.y = y
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.y)

class ToyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        from sklearn.datasets import load_iris
        data = load_iris()
        x = torch.tensor(data["data"], dtype=torch.float32)
        y = torch.tensor(data["target"], dtype=torch.float32).reshape(-1, 1)
        train_idx, valid_idx, test_idx = 40, 45, 50
        if stage == "fit":
            self.train_dataset = ToyDataset(torch.cat((x[:train_idx], x[50:50+train_idx]), dim=0), torch.cat((y[:train_idx], y[50:50+train_idx]), dim=0))
            self.valid_dataset = ToyDataset(torch.cat((x[train_idx:valid_idx], x[50+train_idx:50+valid_idx]), dim=0), torch.cat((y[train_idx:valid_idx], y[50+train_idx:50+valid_idx]), dim=0))
        elif stage == "test":
            self.test_dataset  = ToyDataset(torch.cat((x[valid_idx:test_idx], x[50+valid_idx:50+test_idx]), dim=0), torch.cat((y[valid_idx:test_idx], y[50+valid_idx:50+test_idx]), dim=0))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,  shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)