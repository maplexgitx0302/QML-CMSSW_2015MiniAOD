# %%
"""
# CMS Jet Data Machine Learning with Arc Kernel
"""

# %%
"""
### Import packages
"""

# %%
import os
import jet
import uproot
import awkward as ak

import pennylane as qml
import pennylane.numpy as np

import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

GPU = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
"""
### Model structure
"""

# %%
"""
##### Classical layers and models
"""

# %%
class ClassicalModel(torch.nn.Module):
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

# %%
"""
##### Quantum layers and models
"""

# %%
def encode_daughter_pt_ratio_delta(inputs, cluster_r=1):
    inputs = inputs.reshape((3, -1))
    num_particles = inputs.shape[1]
    num_qubits = 3 * num_particles
    norm_pt, norm_eta, norm_phi = inputs
    for ptc in range(num_particles):
        qml.RY(2 * torch.asin(norm_pt[ptc]), wires=3*ptc)
        qml.RY(2 * torch.asin(norm_pt[ptc]), wires=3*ptc+1)
        qml.CRY(2 * torch.asin(norm_eta[ptc]), wires=[3*ptc, 3*ptc+2])
        qml.CRY(2 * torch.asin(norm_phi[ptc]), wires=[3*ptc+1, 3*ptc+2])

def qml_torch_layer(num_qubits, weight_shapes, enc_layer, qml_layer):
    dev = qml.device('default.qubit', wires=num_qubits)
    @qml.qnode(dev)
    def qnode(inputs, weights):
        enc_layer(inputs)
        qml_layer(weights=weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
    return qml.qnn.TorchLayer(qnode, weight_shapes)

class HybridModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, num_qubits, weight_shapes, enc_layer, qml_layer):
        super().__init__()
        self.num_particles = num_qubits // 3
        self.qkernel = qml_torch_layer(num_qubits, weight_shapes, enc_layer, qml_layer)
        self.net = ClassicalModel(input_dim+num_qubits, hidden_dim, hidden_layers)
    def forward(self, x):
        num_particles = self.num_particles
        norm_pt_eta_phi = x[:, -3*num_particles:].reshape(-1, 3, num_particles)
        norm_pt, norm_eta, norm_phi = norm_pt_eta_phi[:, 0], norm_pt_eta_phi[:, 1], norm_pt_eta_phi[:, 2]
        circuit_input = torch.cat((norm_pt, norm_eta, norm_phi), dim=1)
        x = torch.cat((x[:, :-3*num_particles], self.qkernel(circuit_input)), dim=1)
        y = self.net(x)
        return y

# %%
"""
### Training procedure
"""

# %%
def train(model, data_loader, config):
    print(f"Training with device : {GPU}")
    model = model.to(GPU)
    loss = nn.BCEWithLogitsLoss(reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    record = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
    for epoch in range(config["num_epochs"]):
        train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
        model.train()
        for x, y_true in tqdm.tqdm(data_loader["train"], desc=f"Train({len(data_loader['train'].dataset)})"):
            x, y_true = x.to(GPU), y_true.to(GPU)
            opt.zero_grad()
            y_pred = model(x)
            batch_loss = loss(y_pred, y_true)
            batch_loss.backward()
            train_loss += batch_loss.detach().cpu()
            # BCEWithLogitsLoss -> >=0 | Sigmoid + BCELoss -> >=0.5
            train_acc += (torch.sum((y_pred >= 0) == y_true) / len(x))
            opt.step()
        model.eval()
        for x, y_true in tqdm.tqdm(data_loader["test"], desc=f"Test({len(data_loader['test'].dataset)})"):
            x, y_true = x.to(GPU), y_true.to(GPU)
            y_pred = model(x)
            batch_loss = loss(y_pred, y_true)
            test_loss += (batch_loss * len(x)).detach().cpu()
            # BCEWithLogitsLoss -> >=0 | Sigmoid + BCELoss -> >=0.5
            test_acc += (torch.sum((y_pred >= 0) == y_true).item())
        train_loss /= len(data_loader["train"])
        train_acc /= len(data_loader["train"])
        test_loss /= len(data_loader["test"].dataset)
        test_acc /= len(data_loader["test"].dataset)
        record["train_loss"].append(train_loss)
        record["train_acc"].append(train_acc)
        record["test_loss"].append(test_loss)
        record["test_acc"].append(test_acc)
        print(f"Epoch {epoch+1} : train = (loss:{train_loss:.2f}, acc:{train_acc:.2f}) | test = (loss:{test_loss:.2f}, acc:{test_acc:.2f})")
    return record

# %%
"""
### Data
"""

# %%
def load_data_buffer(channel, get_method, *args):
    suffix = " ".join(map(str, args))
    buffer_file = f"data_buffer/{channel}-{get_method.__name__}-{suffix}.pt"
    if not os.path.exists(buffer_file):
        print(f"Buffer ({get_method.__name__}) : {channel}.pt not found, create now ...")
        events = get_method(channel, *args)
        torch.save(events, buffer_file)
    else:
        events = torch.load(buffer_file)
        print(f"Buffer ({get_method.__name__}) : {channel}.pt found, loading complete!")
    return events

def get_events(channel, num_events, num_particles, jet_type, cut):
    jet_parent = load_data_buffer(channel, jet.get_parent_info, num_events, jet_type, cut)
    if num_particles >= 1:
        jet_daughter = load_data_buffer(channel, jet.get_daughter_info, num_events, num_particles, jet_type, cut)
        return torch.cat((jet_parent, jet_daughter), dim=1)
    else:
        return jet_parent

jet_type   = "fatjet"
num_events = 50000
num_particles = 3
cut = f"({jet_type}_pt >= 500) & ({jet_type}_pt <= 1500)"

signal_channel = "ZprimeToZhToZinvhbb"
# signal_channel = "ZprimeToZhToZlephbb"
# background_channel = "QCD_HT1500to2000"
background_channel = "QCD_HT2000toInf"

data_ratio = 0.9
signal_events = get_events(signal_channel, num_events, num_particles, jet_type, cut)
background_events = get_events(background_channel, num_events, num_particles, jet_type, cut)
num_sig, num_bkg = len(signal_events), len(background_events)
num_data = min(num_sig, num_bkg)
num_train = int(data_ratio * num_data)
num_test = num_data - num_train
print("-" * 100)
print(f"Signal = {signal_channel} | Background = {background_channel} | Cut = {cut}")
print(f"Length Signal = {num_sig} | Length Background = {num_bkg} | Number of Data = {num_data} | Shape = {signal_events.shape}")
print(f"number of training data = {num_train} | number of testing data = {num_test}")

# %%
class JetDataset(Dataset):
    def __init__(self, signal_events, background_events, num_particles, norm):
        x = torch.cat((signal_events ,background_events), dim=0)
        y = torch.cat((torch.ones((len(signal_events)), 1), torch.zeros((len(background_events)), 1)), dim=0)
        # add norm_pt, norm_eta, norm_phi
        if num_particles > 0 and norm == True:
            parent_pt_eta_phi = x[:, :3].reshape(-1, 3, 1)
            parent_pt, parent_eta, parent_phi = parent_pt_eta_phi[:, 0], parent_pt_eta_phi[:, 1], parent_pt_eta_phi[:, 2]
            daughter_pt_eta_phi = x[:, -num_particles*3:].reshape(-1, 3, num_particles)
            daughter_pt, daughter_eta, daughter_phi = daughter_pt_eta_phi[:, 0], daughter_pt_eta_phi[:, 1], daughter_pt_eta_phi[:, 2]
            pt_ratio, delta_eta, delta_phi = daughter_pt/parent_pt, daughter_eta-parent_eta, daughter_phi-parent_phi
            delta_r, cluster_radius = torch.sqrt(delta_eta**2 + delta_phi**2), 1
            norm_pt  = pt_ratio * delta_r / cluster_radius
            norm_eta = delta_eta / delta_r
            norm_phi = delta_phi / delta_r
            if not ((torch.abs(norm_pt) <= 1).all() and (torch.abs(norm_eta) <= 1).all() and  (torch.abs(norm_phi) <= 1).all()):
                num_norm_pt  = torch.sum(torch.abs(norm_pt) > 1)
                num_norm_eta = torch.sum(torch.abs(norm_eta) > 1)
                num_norm_phi = torch.sum(torch.abs(norm_phi) > 1)
                print(f"Recieve value greater than 1 in torch.asin() : (pt={num_norm_pt}, eta={num_norm_eta}, phi={num_norm_phi})")
                if num_norm_pt > 0:
                    norm_pt[norm_pt > 1] = 1
                    norm_pt[norm_pt < -1] = -1
                if num_norm_eta > 0:
                    norm_eta[norm_eta > 1] = 1
                    norm_eta[norm_eta < -1] = -1
                if num_norm_phi > 0:
                    norm_phi[norm_phi > 1] = 1
                    norm_phi[norm_phi < -1] = -1
            x = torch.cat((x, norm_pt, norm_eta, norm_phi), dim=-1)
        x.requires_grad = False
        y.requires_grad = False
        self.x, self.y = x, y
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.y)

c_data_train = JetDataset(signal_events[:num_train], background_events[:num_train], num_particles, norm=False)
c_data_test = JetDataset(signal_events[num_train:num_data], background_events[num_train:num_data], num_particles, norm=False)
q_data_train = JetDataset(signal_events[:num_train], background_events[:num_train], num_particles, norm=True)
q_data_test = JetDataset(signal_events[num_train:num_data], background_events[num_train:num_data], num_particles, norm=True)

# %%
"""
### Result
"""

# %%
cf = {
    "learning_rate":1E-3,
    "weight_decay":0,
    "num_epochs":30,
    "batch_size":32,
}

c_data_loader = {
    "train":DataLoader(c_data_train, cf["batch_size"], shuffle=True, drop_last=True),
    "test":DataLoader(c_data_test, cf["batch_size"], shuffle=False, drop_last=False),
    }
q_data_loader = {
    "train":DataLoader(q_data_train, cf["batch_size"], shuffle=True, drop_last=True),
    "test":DataLoader(q_data_test, cf["batch_size"], shuffle=False, drop_last=False),
    }

input_dim = signal_events.shape[1]
hidden_dim, hidden_layers = 100 * input_dim, 2
num_qubits = 3 * num_particles
weight_shapes = {"weights" : (2, num_qubits)}
enc_layer = encode_daughter_pt_ratio_delta
qml_layer = qml.BasicEntanglerLayers

suffix = f"{jet_type}_{signal_channel}_vs_{background_channel}_{''.join(cut.split())}_"
suffix += f"stc_id{input_dim}_hd{hidden_dim}_hl{hidden_layers}_"
suffix += f"qw_{weight_shapes['weights']}_np{num_particles}_nq{num_qubits}"

c_train_mode = True
q_train_mode = True
if c_train_mode:
    c_model = ClassicalModel(input_dim, hidden_dim, hidden_layers)
    c_result = train(c_model, c_data_loader, cf)
    np.save(f"result_arckernel/c_{suffix}.npy", c_result)
if q_train_mode:
    q_model = HybridModel(input_dim, hidden_dim, hidden_layers, num_qubits, weight_shapes, enc_layer, qml_layer)
    q_result = train(q_model, q_data_loader, cf)
    np.save(f"result_arckernel/q_{suffix}.npy", q_result)