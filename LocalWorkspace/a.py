# %%
"""
## Setup
"""

# %%
"""
### Packages

- `awkward`: For dealing with nested, variable-sized data.
- `pennylane`: Quantum machine learning.
- `lightning`: Simplifying training process.
- `pytorch_geometric`: Graph neural network package.
- `wandb`: Monitoring training process.
"""

# %%
# basic packages
import os, time, random
from itertools import product
import matplotlib.pyplot as plt

# data
import awkward as ak
import d_hep_data

# qml
import pennylane as qml
from pennylane import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# pytorch_lightning
import lightning as L
import lightning.pytorch as pl

# pytorch_geometric
import networkx as nx
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

# scipy
from sklearn import metrics

# wandb
import wandb
from lightning.pytorch.loggers import WandbLogger
wandb.login()

# reproducibility
L.seed_everything(3020616)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# faster calculation on GPU but less precision
torch.set_float32_matmul_precision("medium")

# %%
"""
### Configurations

Hyperparameters and configurations for:
- Data (channel, .etc)
- Training process (Trainer, .etc)
- Model architecture (input/output dimension, .etc)
"""

# %%
# configuration dictionary
cf = {}
cf["time"]     = time.strftime("%Y%m%d_%H%M%S", time.localtime())
cf["wandb"]    = True
cf["project"]  = "g_4vec_2pcqnn"
cf["rnd_seed"] = None # to be determined by for loop

# data infotmation
cf["num_events"]    = "50000"
cf["sig_channel"]   = "ZprimeToZhToZinvhbb"
cf["bkg_channel"]   = "QCD_HT2000toInf"
cf["jet_type"]      = "fatjet"
cf["subjet_radius"] = None # to be determined from [0.25, 0.5, 0.75]
cf["cut_limit"]     = (500, 1500)
cf["bin"]           = 10
cf["num_bin_data"]  = None # to be determined from [100, 200, 300]

# traning configuration
cf["num_train_ratio"]   = 0.5
cf["num_test_ratio"]    = 0.5
cf["batch_size"]        = 64
cf["num_workers"]       = 0
cf["max_epochs"]        = 100
cf["accelerator"]       = "cpu"
cf["num_data"]          = None # to be determined = bin * num_bin_data
cf["fast_dev_run"]      = False
cf["log_every_n_steps"] = cf["batch_size"] // 2

# model hyperparameters
cf["loss_function"]  = nn.BCEWithLogitsLoss()
cf["optimizer"]      = optim.Adam
cf["learning_rate"]  = 1E-3

# 2PCNN hyperparameters
cf["gnn_layers"] = 1
cf["mlp_layers"] = 2

# %%
"""
## Data Module
"""

# %%
"""
In this project, we train with data containing only the four momentum of particles. In order to reduce the size of the data (due to the long training time for quantum machine learning), we reduce the size of data by `fastjet` package by clustering particles again by `anti-kt algorithm` with smaller radius.

The detail (source code) for creating fastjet reclustering events is in the `d_hep_data` file.

To test the power of QML for learning space structure of data (geometric angles, e.g. $p_t$, $\eta$, $\phi$), we will use four momentum only (or z-boosted invariant variables $p_t$, $\eta$, $\phi$).
"""

# %%
def events_pt_eta_phi(events, norm):
    if norm:
        f1 = np.arctan(events["_pt"] / events["pt"])
        f2 = np.pi / 2 * events["_delta_eta"]
        f3 = np.pi * events["_delta_phi"]
        arrays = ak.zip([f1, f2, f3])
    else:
        f1 = events["_pt"]
        f2 = events["_delta_eta"]
        f3 = events["_delta_phi"]
        arrays = ak.zip([f1, f2, f3])
    arrays = arrays.to_list()
    x = [torch.tensor(arrays[i], dtype=torch.float32) for i in range(len(arrays))]
    return x

# %%
class JetDataModule(pl.LightningDataModule):
    def __init__(self, events_func, **kwargs):
        '''Add a "_" prefix if it is a fastjet feature'''
        super().__init__()
        # jet events
        arg_events = [cf["num_events"], cf["jet_type"], cf["subjet_radius"], cf["cut_limit"], cf["bin"], cf["num_bin_data"]]
        sig_events = d_hep_data.events_uniform_Pt_weight(cf["sig_channel"], *arg_events)
        sig_events = events_func(sig_events, **kwargs)
        bkg_events = d_hep_data.events_uniform_Pt_weight(cf["bkg_channel"], *arg_events)
        bkg_events = events_func(bkg_events, **kwargs)
        self.sig_data_list = self._create_data_list(sig_events, 1)
        self.bkg_data_list = self._create_data_list(bkg_events, 0)

        # count the number of training, and testing
        assert len(self.sig_data_list) >= cf["num_data"], f"sig data not enough: {len(self.sig_data_list)} < {cf['num_data']}"
        assert len(self.bkg_data_list) >= cf["num_data"], f"bkg data not enough: {len(self.bkg_data_list)} < {cf['num_data']}"
        num_train = int(cf["num_data"] * cf["num_train_ratio"])
        num_test  = int(cf["num_data"] * cf["num_test_ratio"])
        print(f"DataLog: {cf['sig_channel']} has {len(self.sig_data_list)} events and {cf['bkg_channel']} has {len(self.bkg_data_list)} events.")
        print(f"Choose num_data for each channel to be {cf['num_data']} | Each channel  has num_train = {num_train}, num_test = {num_test}")

        # prepare dataset for dataloader
        train_idx = num_train
        test_idx  = num_train + num_test
        self.train_dataset = self.sig_data_list[:train_idx] + self.bkg_data_list[:train_idx]
        self.test_dataset  = self.sig_data_list[train_idx:test_idx] + self.bkg_data_list[train_idx:test_idx]
    
    def _create_data_list(self, events, y):
        # create pytorch_geometric "Data" object
        data_list = []
        for i in range(len(events)):
            x = events[i]
            edge_index = list(product(range(len(x)), range(len(x))))
            edge_index = torch.tensor(edge_index).transpose(0, 1)
            x.requires_grad, edge_index.requires_grad = False, False
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        random.shuffle(data_list)
        return data_list

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=cf["batch_size"], num_workers=cf["num_workers"],  shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=cf["batch_size"], num_workers=cf["num_workers"])

# %%
"""
## Models
"""

# %%
"""
To compare classical GNN with quantum GNN, we use `GraphConv` and `MessagePassing` with `pennylane` for classical and quantum repectively.

- Why using `nn.ModuleList` instead of `nn.Sequential`?
Both `nn.ModuleList` and `nn.Sequential` trace the trainable parameters autometically. However, since we are using "gnn" layers, we need to feed into additional argument `edge_index`. In order to check whether we are using "gnn" layers or not, we use `isinstance` to check the class type (Since all PyTorch Geometric graph layer inherit the class `MessagePassing`). For detail, see [When should I use nn.ModuleList and when should I use nn.Sequential?](https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/3)
"""

# %%
"""
### MLP Layers
"""

# %%
class ClassicalMLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel, num_layers):
        super().__init__()
        if num_layers == 0:
            self.net = nn.Linear(in_channel, out_channel)
        else:
            net = [nn.Linear(in_channel, hidden_channel), nn.ReLU()]
            for _ in range(num_layers-2):
                net += [nn.Linear(hidden_channel, hidden_channel), nn.ReLU()]
            net += [nn.Linear(hidden_channel, out_channel)]
            self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)
    
class QuantumMLP(nn.Module):
    def __init__(self, num_qubits, num_layers, num_reupload, measurements):
        super().__init__()
        # create a quantum MLP
        @qml.qnode(qml.device('lightning.qubit', wires=num_qubits), diff_method="adjoint")
        def circuit(inputs, weights):
            for i in range(num_reupload):
                qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights=weights[i], wires=range(num_qubits))
            measurements_dict = {"X":qml.PauliX, "Y":qml.PauliY, "Z":qml.PauliZ}
            return [qml.expval(measurements_dict[m[1]](wires=m[0])) for m in measurements]
        # turn the quantum circuit into a torch layer
        weight_shapes = {"weights":(num_reupload, num_layers, num_qubits, 3)}
        net = [qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)]
        self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)

# %%
"""
### Classical 2PCNN
"""

# %%
class Classical2PCNNForwardMP(MessagePassing):
    def __init__(self, num_features, num_layers, aggr):
        super().__init__(aggr=aggr)
        self.mp_phi = ClassicalMLP(
            in_channel     = 2*num_features,
            out_channel    = num_features,
            hidden_channel = 2*num_features,
            num_layers     = num_layers,
            )
        self.mp_gamma = ClassicalMLP(
            in_channel     = 2*num_features, 
            out_channel    = num_features,
            hidden_channel = 2*num_features,
            num_layers     = num_layers,
        )
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.mp_phi(torch.cat((x_i, x_j), dim=-1))
    def update(self, aggr_out, x):
        return self.mp_gamma(torch.cat((x, aggr_out), dim=-1))

class Classical2PCNNForward(nn.Module):
    def __init__(self, gnn_in, gnn_layers, gnn_aggr, mlp_in, mlp_out, mlp_hidden, mlp_layers):
        super().__init__()
        self.gnn = Classical2PCNNForwardMP(gnn_in, gnn_layers, gnn_aggr)
        self.mlp = ClassicalMLP(mlp_in, mlp_out, mlp_hidden, mlp_layers)
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch)
        x = self.mlp(x)
        return x

# %%
"""
### Quantum 2PCQNN
"""

# %%
class Quantum2PCQNNForwardMP(MessagePassing):
    def __init__(self, num_features, num_layers, num_reupload, aggr):
        super().__init__(aggr=aggr)
        measurements = [[i, "Z"] for i in range(num_features)]
        self.mp_phi = QuantumMLP(
            num_qubits   = 2*num_features, 
            num_layers   = num_layers,
            num_reupload = num_reupload,
            measurements = measurements,
            )
        self.mp_gamma = ClassicalMLP(
            in_channel     = 2*len(measurements), 
            out_channel    = num_features,
            hidden_channel = 2*num_features,
            num_layers     = num_layers,
        )
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.mp_phi(torch.cat((x_i, x_j), dim=-1))
    def update(self, aggr_out, x):
        return self.mp_gamma(torch.cat((x, aggr_out), dim=-1))

class Quantum2PCQNNForward(nn.Module):
    def __init__(self, gnn_in, gnn_layers, gnn_reupload, gnn_aggr, mlp_in, mlp_out, mlp_hidden, mlp_layers):
        super().__init__()
        self.gnn = Quantum2PCQNNForwardMP(gnn_in, gnn_layers, gnn_reupload, gnn_aggr)
        self.mlp = ClassicalMLP(mlp_in, mlp_out, mlp_hidden, mlp_layers)
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch)
        x = self.mlp(x)
        return x

# %%
"""
### Lightning Module

Most of the hyperparameters are defined at `cf` configuration dictionary.

Note that when using `nn.BCEWithLogitsLoss`, the first argument should not be paased to `sigmoid`.
"""

# %%
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_function = cf["loss_function"]

    def forward(self, data):
        # predict y
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch)
        x = x.squeeze(dim=-1)

        # calculate loss and accuracy
        y_pred = x > 0
        y_true = data.y
        loss   = self.loss_function(x, y_true.float())
        acc    = (y_pred == data.y).float().mean()

        # calculate auc
        y_score = torch.sigmoid(x).detach()
        self.y_true_buffer  = torch.cat((self.y_true_buffer, y_true))
        self.y_score_buffer = torch.cat((self.y_score_buffer, y_score))
        return loss, acc

    def configure_optimizers(self):
        optimizer = cf["optimizer"](self.parameters(), lr=cf["learning_rate"])
        return optimizer

    def on_train_epoch_start(self):
        self.start_time     = time.time()
        self.y_true_buffer  = torch.tensor([])
        self.y_score_buffer = torch.tensor([])

    def on_train_epoch_end(self):
        self.end_time = time.time()
        delta_time = self.end_time - self.start_time
        roc_auc    = metrics.roc_auc_score(self.y_true_buffer, self.y_score_buffer)
        self.log("epoch_time", delta_time, on_step=False, on_epoch=True)
        self.log("train_roc_auc", roc_auc, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self.y_true_buffer  = torch.tensor([])
        self.y_score_buffer = torch.tensor([])

    def on_test_epoch_end(self):
        roc_auc = metrics.roc_auc_score(self.y_true_buffer, self.y_score_buffer)
        self.log("test_roc_auc", roc_auc, on_step=False, on_epoch=True)

    def training_step(self, data, batch_idx):
        loss, acc = self.forward(data)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(data.x))
        self.log("train_acc", acc, on_step=True, on_epoch=True, batch_size=len(data.x))
        return loss

    def test_step(self, data, batch_idx):
        _, acc = self.forward(data)
        self.log("test_acc", acc, on_step=True, on_epoch=True, batch_size=len(data.x))

# %%
"""
## Train/Test the Model
"""

# %%
"""
### Training procedure
"""

# %%
def train(model, data_module, commit="", suffix=""):
    # setup id and path for saving
    project  = cf['project']
    group    = f"{cf['time']}_{commit}_{cf['sig_channel']}_{cf['bkg_channel']}_{cf['jet_type']}"
    job_type = f"R{cf['subjet_radius']}_D{cf['num_data']}"
    name     = f"{model.__class__.__name__}_{suffix} | {job_type} | {group} | {cf['rnd_seed']}"
    id       = f"{name}"
    tags     = [model.__class__.__name__, cf['sig_channel'], cf['bkg_channel'], cf['jet_type'], str(cf['subjet_radius']), str(cf['num_data'])]
    root_dir = f"./result"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    # wandb logger setup
    if cf["wandb"]:
        cf["group_rnd_seed"] = f"{model.__class__.__name__}_{suffix} | {job_type}"
        cf["model_name"]     = model.__class__.__name__
        wandb_logger = WandbLogger(project=project, group=group, job_type=job_type, name=name, id=id, save_dir=root_dir, tags=tags)
        wandb_logger.experiment.config.update(cf)
        wandb_logger.watch(model, log="all")

    # start lightning training
    logger   = wandb_logger if cf["wandb"] else None
    trainer  = L.Trainer(
        logger=logger, 
        accelerator       = cf["accelerator"],
        max_epochs        = cf["max_epochs"],
        fast_dev_run      = cf["fast_dev_run"],
        log_every_n_steps = cf["log_every_n_steps"],
        )
    litmodel = LitModel(model)
    trainer.fit(litmodel, datamodule=data_module)
    trainer.test(litmodel, datamodule=data_module)

    # finish wandb monitoring
    if cf["wandb"]:
        wandb.finish()

# %%
"""
### Start training each model
"""

# %%
def grid_train(rnd_seed, num_bin_data, R, commit=""):
    # setup
    L.seed_everything(rnd_seed)
    cf["subjet_radius"] = R
    cf["rnd_seed"]      = rnd_seed
    cf["num_bin_data"]  = num_bin_data
    cf["num_data"]      = cf["bin"] * cf["num_bin_data"]

    # data module
    data_pt_eta_phi = JetDataModule(events_func=events_pt_eta_phi, norm=False)
    data_pt_eta_phi_norm = JetDataModule(events_func=events_pt_eta_phi, norm=True)

    # classical 2pcnn
    input_dim = data_pt_eta_phi.train_dataset[0].x.shape[1]
    cf_2pcnn = {
        "gnn_in":input_dim, 
        "gnn_layers":cf["gnn_layers"],
        "gnn_aggr":"add", 
        "mlp_in":input_dim,
        "mlp_out":1, 
        "mlp_hidden":3*input_dim, 
        "mlp_layers":cf["mlp_layers"],
    }
    train(Classical2PCNNForward(**cf_2pcnn), data_pt_eta_phi, commit, suffix=f"pep")
    train(Classical2PCNNForward(**cf_2pcnn), data_pt_eta_phi_norm, commit, suffix=f"pep_norm")

    # quantum 2pcqnn
    input_dim = data_pt_eta_phi_norm.train_dataset[0].x.shape[1]
    cf_2pcqnn = {
        "gnn_in":input_dim,
        "gnn_layers":cf["gnn_layers"],
        "gnn_reupload":input_dim,
        "gnn_aggr":"add",
        "mlp_in":input_dim,
        "mlp_out":1,
        "mlp_hidden":3*input_dim,
        "mlp_layers":cf["mlp_layers"],
    }
    # train(Quantum2PCQNNForward(**cf_2pcqnn), data_pt_eta_phi_norm, commit, suffix=f"pep_norm")

commit = input("Commit of this experiment (short description): ")
for R in [0.75, 0.5, 0.25]:
    for num_bin_data in [100, 200, 400]:
        for rnd_seed in range(3):
            grid_train(rnd_seed, num_bin_data, R, commit)