# %%
"""
### Import packages
"""

# %%
# system
import os, glob
import time, datetime
from tqdm import tqdm

# hep and toy model
import toy
import uproot
import awkward as ak
import hep_events, hep_buffer

# qml
import pennylane as qml
import pennylane.numpy as np

# pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# pytorch_lightning
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# wandb
import wandb
from pytorch_lightning.loggers import WandbLogger
wandb.login()

# %%
"""
### Hyper-parameters
"""

# %%
project_name = "wandb_arckernel"
jet_type = "jet"

# whether use toy data and quantum model
train_toy = False
train_quantum = True
enable_logger = True
fast_dev_run = False
if train_toy:
    project_name = "wandb_toy"

# hep data hyper-parameters
data_config = {
    "num_events":50000,
    "num_particles":3,
    "cut":f"({jet_type}_pt>=500)&({jet_type}_pt<=1500)",
    "signal_channel":"ZprimeToZhToZinvhbb",
    "background_channel":"QCD_HT2000toInf",
}

# hyper-parameters
hyper_config = {
    # general settings
    "random_seed":0,
    "max_num_data":2000,
    "train_ratio":0.8,
    "valid_ratio":0.2,

    # constants
    "lr":1E-3,
    "batch_size":128,
    "num_workers":12,

    # functions
    "logits_threshold":0.5,
    "loss_function":F.binary_cross_entropy,
    "accuracy_function":torchmetrics.Accuracy(task="binary"),

    # wandb
    "log_freq":1,
    }

# pytorch_lightning trainer
trainer_config = {
    "accelerator":"gpu",
    "max_epochs":60,
    "fast_dev_run":fast_dev_run,
    "deterministic":True,
    "log_every_n_steps":1,
    "auto_scale_batch_size":None,
}

# for trainer callbacks config (need to reinitiate callback for each run)
callback_config = {
    EarlyStopping:{"monitor":"train_loss", "min_delta":0, "mode":"min", "patience":10}
}

if train_toy:
    hyper_config, trainer_config, ansatz_config = toy.set_config(hyper_config, trainer_config)
else:
    if jet_type == "jet":
        input_dim = 3 + 3*data_config["num_particles"] # pt eta phi
    elif jet_type == "fatjet":
        input_dim = 3 + 3*data_config["num_particles"] + 5 # pt eta phi and n-subjettiness
    # ansatz
    ansatz_config = [
        # input_dim, hidden_dim, hidden_layers, num_layers, num_reupload
        (input_dim, 10*input_dim, 3, 0, 0),
        (input_dim, 10*input_dim, 4, 0, 0),
        (input_dim, 10*input_dim, 5, 0, 0),
        (input_dim, 10*input_dim, 6, 0, 0),
        (input_dim, 50*input_dim, 3, 0, 0),
        (input_dim, 50*input_dim, 4, 0, 0),
        (input_dim, 50*input_dim, 5, 0, 0),
        (input_dim, 50*input_dim, 6, 0, 0),

        (input_dim, 1*input_dim, 3, 1, 1),
        (input_dim, 1*input_dim, 4, 1, 1),
        (input_dim, 5*input_dim, 3, 1, 1),
        (input_dim, 5*input_dim, 4, 1, 1),
        (input_dim, 1*input_dim, 3, 1, 2),
        (input_dim, 1*input_dim, 4, 1, 2),
        (input_dim, 5*input_dim, 3, 1, 2),
        (input_dim, 5*input_dim, 4, 1, 2),
    ]

# %%
"""
### Data
"""

# %%
def get_data(channel):
    num_events, num_particles, cut = data_config["num_events"], data_config["num_particles"], data_config["cut"]
    jet_parent = hep_buffer.load_data_buffer(channel, hep_buffer.get_parent_info, num_events, jet_type, cut)
    if num_particles >= 1:
        jet_daughter = hep_buffer.load_data_buffer(channel, hep_buffer.get_daughter_info, num_events, num_particles, jet_type, cut)
        return torch.cat((jet_parent, jet_daughter), dim=1)
    else:
        return jet_parent

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
        daughter_pt_eta_phi = events[:, -3*data_config["num_particles"]:].reshape(-1, 3, data_config["num_particles"])
        parent_pt, parent_eta, parent_phi = parent_pt_eta_phi.transpose(0, 1)
        daughter_pt, daughter_eta, daughter_phi = daughter_pt_eta_phi.transpose(0, 1)
        pt_ratio, delta_eta, delta_phi = daughter_pt/torch.sum(daughter_pt**2), daughter_eta-parent_eta, daughter_phi-parent_phi
        delta_r, cluster_radius = torch.sqrt(delta_eta**2 + delta_phi**2), 1
        norm_pt  = (pt_ratio) * (delta_r / cluster_radius)
        norm_eta = (delta_eta / delta_r)
        norm_phi = (delta_phi / delta_r)
        if not ((torch.abs(norm_pt) <= 1).all() and (torch.abs(norm_eta) <= 1).all() and  (torch.abs(norm_phi) <= 1).all()):
            num_norm_pt  = torch.sum(torch.abs(norm_pt) > 1)
            num_norm_eta = torch.sum(torch.abs(norm_eta) > 1)
            num_norm_phi = torch.sum(torch.abs(norm_phi) > 1)
            raise(ValueError(f"Recieve value greater than pi in torch.asin() : (num_pt={num_norm_pt}, num_eta={num_norm_eta}, num_phi={num_norm_phi})"))
        else:
            print(f"Log(get_norm):Arguments (norms) of arcsin are within [-1, 1]")
        events = torch.cat((events, norm_pt, norm_eta, norm_phi), dim=-1)
        return events

class JetDataModule(pl.LightningDataModule):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
        # get signal and background data
        self.signal_events     = get_data(data_config["signal_channel"])
        self.background_events = get_data(data_config["background_channel"])
        # for auto batch finding
        self.batch_size = hyper_config["batch_size"]
        self.num_workers = hyper_config["num_workers"]
        # determine num_train, num_valid, num_test
        num_sig   = len(self.signal_events)
        num_bkg   = len(self.background_events)
        num_data  = min(num_sig, num_bkg, hyper_config["max_num_data"])
        num_train = int(hyper_config["train_ratio"] * num_data)
        num_valid = int(hyper_config["valid_ratio"] * num_train)
        num_test  = num_data - num_train
        num_train = num_train - num_valid
        self.num_train, self.num_valid, self.num_test = num_train, num_valid, num_test
        print(f"JetDataModule INFO: num_train = {num_train}, num_valid = {num_valid}, num_test = {num_test}")

    def setup(self, stage):
        train_idx = self.num_train
        valid_idx = self.num_train + self.num_valid
        test_idx  = self.num_train + self.num_valid + self.num_test
        if stage == "fit":
            self.train_dataset = JetDataset(self.signal_events[:train_idx], self.background_events[:train_idx], norm=self.norm)
            self.valid_dataset = JetDataset(self.signal_events[train_idx:valid_idx], self.background_events[train_idx:valid_idx], norm=self.norm)
        elif stage == "test":
            self.test_dataset  = JetDataset(self.signal_events[valid_idx:test_idx], self.background_events[valid_idx:test_idx], norm=self.norm)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,  shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# %%
"""
### Classical Model
"""

# %%
class ClassicalFNNModel(nn.Module):
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
        net += [nn.Sigmoid()]
        self.net = nn.Sequential(*net)
    def forward(self, x):
        y = self.net(x)
        return y

# %%
"""
### Quantum Model
"""

# %%
"""
##### Encoding layers (ENC) and Variational Quantum Circuit (VQC)
"""

# %%
class ENCDaughterNorm:
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.num_qubits = 3 * num_particles
    def __call__(self, inputs):
        inputs = inputs.reshape((3, -1))
        norm_pt, norm_eta, norm_phi = inputs
        for ptc in range(self.num_particles):
            qml.RY(2 * torch.asin(norm_pt[ptc]), wires=3*ptc)
            qml.RY(2 * torch.asin(norm_pt[ptc]), wires=3*ptc+1)
            qml.CRY(2 * torch.asin(norm_eta[ptc]), wires=[3*ptc, 3*ptc+2])
            qml.CRY(2 * torch.asin(norm_phi[ptc]), wires=[3*ptc+1, 3*ptc+2])

class VQCRotCNOT:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
    def __call__(self, weights):
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        for l in range(num_layers):
            for q in range(num_qubits):
                qml.Rot(*weights[l][q], wires=q)
            for q in range(num_qubits):
                if q != num_qubits-1:
                    qml.CNOT(wires=[q, q+1])
                else:
                    if num_qubits >= 3:
                        qml.CNOT(wires=[q, 0])

# %%
"""
##### Quantum Layers and Models
"""

# %%
def qml_torch_layer(enc_layer, vqc_layer, weight_shapes, num_reupload):
    num_qubits = max(enc_layer.num_qubits, vqc_layer.num_qubits)
    dev = qml.device('default.qubit', wires=num_qubits)
    @qml.qnode(dev)
    def qnode(inputs, weights):
        for r in range(num_reupload):
            enc_layer(inputs)
            vqc_layer(weights[r])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
    return qml.qnn.TorchLayer(qnode, weight_shapes)


class HybridArcKernelDaughterModel(nn.Module):
    def __init__(self, classical_model, num_particles, num_layers, num_reupload):
        super().__init__()
        self.num_particles = num_particles
        num_qubits = 3 * num_particles
        self.num_qubits = num_qubits
        weight_shapes = {"weights":(num_reupload, num_layers, num_qubits, 3)}
        enc_layer = ENCDaughterNorm(num_particles)
        vqc_layer = VQCRotCNOT(num_qubits, num_layers)
        self.quantum_kernel = qml_torch_layer(enc_layer, vqc_layer, weight_shapes, num_reupload)
        self.classical_model = classical_model
    def forward(self, x):
        num_particles = self.num_particles
        norm_pt_eta_phi = x[:, -3*num_particles:].reshape(-1, 3, num_particles)
        norm_pt, norm_eta, norm_phi = norm_pt_eta_phi[:, 0], norm_pt_eta_phi[:, 1], norm_pt_eta_phi[:, 2]
        quantum_input = torch.cat((norm_pt, norm_eta, norm_phi), dim=1)
        x = torch.cat((x[:, :-3*num_particles], self.quantum_kernel(quantum_input)), dim=1)
        y = self.classical_model(x)
        return y

# %%
"""
### LightningModule
"""

# %%
class LitModel(pl.LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_func = hyper_config["loss_function"]
        self.acc_func = hyper_config["accuracy_function"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        train_loss = self.loss_func(logits, y) # note the order of arg of loss func
        train_acc = self.acc_func(y, logits > hyper_config["logits_threshold"])
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        valid_loss = self.loss_func(logits, y) # note the order of arg of loss func
        valid_acc = self.acc_func(y, logits > hyper_config["logits_threshold"])
        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
        self.log("valid_acc", valid_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        test_loss = self.loss_func(logits, y) # note the order of arg of loss func
        test_acc = self.acc_func(y, logits > hyper_config["logits_threshold"])
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)
        self.log("test_acc", test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hyper_config["lr"])
        return optimizer

# %%
"""
### Training
"""

# %%
def train(model, datamodule, logger_name):
    pl.seed_everything(hyper_config["random_seed"], workers=True)

    # pl logger setup
    logger = None
    if enable_logger:
        logger = WandbLogger(project=project_name, name=f"{logger_name}", id=logger_name)
        wandb_config = {}
        wandb_config.update(data_config)
        wandb_config.update(hyper_config)
        wandb_config.update(trainer_config)
        logger.experiment.config.update(wandb_config, allow_val_change=True)
        logger.watch(model, log="all", log_freq=hyper_config["log_freq"])
    
    # pl callbacks setup
    callbacks = []
    for key, value in callback_config.items():
        callbacks.append(key(**value))

    # trainer setup
    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=callbacks)
    if trainer_config["auto_scale_batch_size"] != None:
        trainer.tune(model, datamodule)
    
    # start fitting and testing
    checkpoints_dir = f"./{project_name}/{logger_name}/checkpoints/"
    try:
        ckpt_path = glob.glob(f'{checkpoints_dir}/*ckpt')[0]
    except:
        ckpt_path = None
        print(f"Log(checkpoints): *.ckpt files not found, start training from initial state.")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.test(model=model, datamodule=datamodule)
    if enable_logger:
        wandb.finish()

# %%
# data modules
pl.seed_everything(hyper_config["random_seed"], workers=True)
if train_toy:
    c_datamodule = toy.ToyDataModule(batch_size=hyper_config["batch_size"], num_workers=hyper_config["num_workers"])
    q_datamodule = toy.ToyDataModule(batch_size=hyper_config["batch_size"], num_workers=hyper_config["num_workers"])
else:
    c_datamodule = JetDataModule(norm=False)
    q_datamodule = JetDataModule(norm=True)

if train_toy:
    for ansatz in tqdm(ansatz_config, "Toy Model"):
        logger_name = f"toy_({','.join(list(map(str, ansatz)))})"
        model = LitModel(ClassicalFNNModel(*ansatz))
        train(model, c_datamodule, logger_name)
else:
    for ansatz in tqdm(ansatz_config, "HEP Model"):
        logger_prefix = f"{jet_type}_ptc{data_config['num_particles']}_n{hyper_config['max_num_data']}_{data_config['cut']}"
        c_logger_name = f"{logger_prefix}_classical_({','.join(list(map(str, ansatz)))})"
        c_model = LitModel(ClassicalFNNModel(*ansatz[:3]))
        train(c_model, c_datamodule, c_logger_name)
        if train_quantum and (ansatz[3] != 0 or ansatz[4] != 0):
            q_logger_name = f"{logger_prefix}_quantum_({','.join(list(map(str, ansatz)))})"
            q_ansatz = list(ansatz)
            q_ansatz[0] += 3 * data_config["num_particles"]
            c_model = ClassicalFNNModel(*q_ansatz[:3])
            q_model = LitModel(HybridArcKernelDaughterModel(c_model, data_config['num_particles'], *q_ansatz[3:]))
            train(q_model, q_datamodule, q_logger_name)