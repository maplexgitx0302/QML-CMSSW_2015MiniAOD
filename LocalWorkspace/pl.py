# %%
"""
### Import packages
"""

# %%
# system
import os

# hep
import uproot
import awkward as ak
import hep_events, hep_buffer

# qml
import pennylane as qml
import pennylane.numpy as np

# pytorch and lightning
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# %%
"""
### Hyper-parameters
"""

# %%
# hep hyper-parameters
jet_type   = "fatjet"
num_events = 50000
num_particles = 3
cut = f"({jet_type}_pt >= 900) & ({jet_type}_pt <= 1100)"
signal_channel = "ZprimeToZhToZinvhbb"
# signal_channel = "ZprimeToZhToZlephbb"
# background_channel = "QCD_HT1500to2000"
background_channel = "QCD_HT2000toInf"

# training hyper-parameters
max_num_data = 2000
train_ratio = 0.8
valid_ratio = 0.2

config = {
    "lr":1E-3,
    "num_epochs":30,
    "batch_size":32,
    "loss_function":nn.BCEWithLogitsLoss(reduction="mean"),
    "fast_dev_run":False,
    "max_epochs":10,
    "num_workers":0,
    }

# %%
"""
### Data
"""

# %%
def get_data(channel, num_events, num_particles, jet_type, cut):
    jet_parent = hep_buffer.load_data_buffer(channel, hep_buffer.get_parent_info, num_events, jet_type, cut)
    if num_particles >= 1:
        jet_daughter = hep_buffer.load_data_buffer(channel, hep_buffer.get_daughter_info, num_events, num_particles, jet_type, cut)
        return torch.cat((jet_parent, jet_daughter), dim=1)
    else:
        return jet_parent

signal_events = get_data(signal_channel, num_events, num_particles, jet_type, cut)
background_events = get_data(background_channel, num_events, num_particles, jet_type, cut)
num_sig, num_bkg = len(signal_events), len(background_events)
num_data = min(num_sig, num_bkg, max_num_data)
num_train = int(train_ratio * num_data)
num_valid = int(valid_ratio * num_train)
num_test = num_data - num_train
num_train = num_train - num_valid
print("-" * 100)
print(f"- Cut = {cut}")
print(f"- Signal = {signal_channel}(num={num_sig})")
print(f"- Background = {background_channel}(num={num_bkg})")
print(f"- Choose number of data = {num_data}(max={max_num_data})")
print(f"- Train={num_train} | Valid={num_valid} | Test={num_test}")
print(f"- Shape of data = {signal_events.shape}")

# %%
class JetDataset(Dataset):
    def __init__(self, signal_events, background_events, norm):
        x = torch.cat((signal_events ,background_events), dim=0)
        if norm: x = self.get_norm(x)
        y = torch.cat((torch.ones((len(signal_events)), 1), torch.zeros((len(background_events)), 1)), dim=0)
        x.requires_grad = False
        y.requires_grad = False
        self.x, self.y = x, y
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.y)
    def get_norm(self, events):
        parent_pt_eta_phi   = events[:, :3].reshape(-1, 3, 1)
        daughter_pt_eta_phi = events[:, -3*num_particles:].reshape(-1, 3, num_particles)
        parent_pt, parent_eta, parent_phi = parent_pt_eta_phi.transpose(0, 1)
        daughter_pt, daughter_eta, daughter_phi = daughter_pt_eta_phi.transpose(0, 1)
        pt_ratio, delta_eta, delta_phi = daughter_pt/parent_pt, daughter_eta-parent_eta, daughter_phi-parent_phi
        delta_r, cluster_radius = torch.sqrt(delta_eta**2 + delta_phi**2), 1
        norm_pt  = (1/daughter_pt) * delta_r / cluster_radius
        norm_eta = delta_eta / delta_r
        norm_phi = delta_phi / delta_r
        if not ((torch.abs(norm_pt) <= 1).all() and (torch.abs(norm_eta) <= 1).all() and  (torch.abs(norm_phi) <= 1).all()):
            num_norm_pt  = torch.sum(torch.abs(norm_pt) > 1)
            num_norm_eta = torch.sum(torch.abs(norm_eta) > 1)
            num_norm_phi = torch.sum(torch.abs(norm_phi) > 1)
            print(f"Recieve value greater than 1 in torch.asin() : (num_pt={num_norm_pt}, num_eta={num_norm_eta}, num_phi={num_norm_phi})")
            if num_norm_pt > 0:
                norm_pt[norm_pt > 1] = 1
                norm_pt[norm_pt < -1] = -1
            if num_norm_eta > 0:
                norm_eta[norm_eta > 1] = 1
                norm_eta[norm_eta < -1] = -1
            if num_norm_phi > 0:
                norm_phi[norm_phi > 1] = 1
                norm_phi[norm_phi < -1] = -1
        events = torch.cat((events, norm_pt, norm_eta, norm_phi), dim=-1)
        return events

# classical dataset
c_train_dataset = JetDataset(signal_events[:num_train], background_events[:num_train], norm=False)
c_valid_dataset = JetDataset(signal_events[num_train:num_train+num_valid], background_events[num_train:num_train+num_valid], norm=False)
c_test_dataset = JetDataset(signal_events[num_train+num_valid:num_data], background_events[num_train+num_valid:num_data], norm=False)

# quantum dataset
norm = num_particles > 0
q_train_dataset = JetDataset(signal_events[:num_train], background_events[:num_train], norm)
q_valid_dataset = JetDataset(signal_events[num_train:num_train+num_valid], background_events[num_train:num_train+num_valid], norm)
q_test_dataset = JetDataset(signal_events[num_train:num_data], background_events[num_train:num_data], norm)

# %%
"""
### Classical Model
"""

# %%
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers):
        super().__init__()
        if hidden_layers == 0:
            net = [nn.Linear(input_dim, 1)]
        else:
            net = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(hidden_layers-1):
                net += [nn.Linear(hidden_dim, hidden_dim)]
                net += [nn.ReLU()]
            net += [nn.Linear(hidden_dim, 1)]
        # BCEWithLogitsLoss already contains a sigmoid function
        self.net = nn.Sequential(*net)
    def forward(self, x):
        y = self.net(x)
        return y

class ClassicalModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        loss = config["loss_function"](y_true, y_pred)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        return optimizer

# %%
"""
### Training
"""

# %%
input_dim = signal_events.shape[1]
hidden_dim, hidden_layers = 100 * input_dim, 2

c_train_loader = DataLoader(c_train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"])

c_nn = FNN(input_dim, hidden_dim, hidden_layers)
c_model = ClassicalModel(c_nn)

trainer = pl.Trainer(max_epochs=config["max_epochs"], fast_dev_run=config["fast_dev_run"], log_every_n_steps=1)
trainer.fit(model=c_model, train_dataloaders=c_train_loader)