# %%
# basic packages
import os, time
from itertools import product
import matplotlib.pyplot as plt

# model template
import m_nn
import m_lightning

# data
import d_hep_data
import awkward as ak

# qml
import pennylane as qml
from pennylane import numpy as np

# pytorch
import torch
import torch.nn as nn

# pytorch_lightning
import lightning as L
import lightning.pytorch as pl

# pytorch_geometric
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

# wandb
import wandb
from lightning.pytorch.loggers import WandbLogger
wandb.login()

# reproducibility
L.seed_everything(3020616)

# faster calculation on GPU but less precision
torch.set_float32_matmul_precision("medium")

# directory for saving results
root_dir = f"./result"
if os.path.isdir(root_dir) == False:
    os.makedirs(root_dir)

# %%
# global settings
cf = {}
cf["time"]     = time.strftime("%Y%m%d_%H%M%S", time.localtime())
cf["wandb"]    = True
cf["project"]  = "g_eflow_2pcgnn"

# traning configuration
cf["num_train_ratio"]   = 0.8
cf["batch_size"]        = 64
cf["num_workers"]       = 0
cf["max_epochs"]        = 50
cf["accelerator"]       = "cpu"
cf["fast_dev_run"]      = False
cf["log_every_n_steps"] = cf["batch_size"] // 2

# %%
class JetDataModule(pl.LightningDataModule):
    def __init__(self, sig_events, bkg_events, mode=None):
        super().__init__()
        # jet events
        sig_events = self._preprocess(sig_events, 1, mode)
        bkg_events = self._preprocess(bkg_events, 0, mode)

        # prepare dataset for dataloader
        train_idx = int(cf["num_train_ratio"] * len(sig_events))
        self.train_dataset = sig_events[:train_idx] + bkg_events[:train_idx]
        self.test_dataset  = sig_events[train_idx:] + bkg_events[train_idx:]

    def _preprocess(self, events, y, mode):
        # "_" prefix means that it is a fastjet feature
        if mode == "normalize":
            f1 = np.arctan(events["_pt"] / events["pt"])
            f2 = events["_delta_eta"]
            f3 = events["_delta_phi"]
        elif mode == "":
            f1 = events["_pt"]
            f2 = events["_delta_eta"]
            f3 = events["_delta_phi"]
        arrays = ak.zip([f1, f2, f3])
        arrays = arrays.to_list()
        events = [torch.tensor(arrays[i], dtype=torch.float32) for i in range(len(arrays))]

        # create pytorch_geometric "Data" object
        data_list = []
        for i in range(len(events)):
            x = events[i]
            edge_index = list(product(range(len(x)), range(len(x))))
            edge_index = torch.tensor(edge_index).transpose(0, 1)
            x.requires_grad, edge_index.requires_grad = False, False
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        return data_list

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=cf["batch_size"], shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=cf["batch_size"], shuffle=False)

# %%
class MessagePassing(MessagePassing):
    def __init__(self, phi):
        super().__init__(aggr="add", flow="target_to_source")
        self.phi = phi
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.phi(torch.cat((x_i, x_j), dim=-1))
    def update(self, aggr_out, x):
        return aggr_out

class Graph2PCGNN(nn.Module):
    def __init__(self, phi, mlp):
        super().__init__()
        self.gnn = MessagePassing(phi)
        self.mlp = mlp
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = geom_nn.global_add_pool(x, batch)
        x = self.mlp(x)
        return x

# %%
class Classical2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_in, gnn_out, gnn_hidden, gnn_layers, mlp_hidden=0, mlp_layers=0):
        phi = m_nn.ClassicalMLP(in_channel=gnn_in, out_channel=gnn_out, hidden_channel=gnn_hidden, num_layers=gnn_layers)
        mlp = m_nn.ClassicalMLP(in_channel=gnn_out, out_channel=1, hidden_channel=mlp_hidden, num_layers=mlp_layers)
        super().__init__(phi, mlp)

class QuantumAngle2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements):
        phi = m_nn.QuantumMLP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements)
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)

class QuantumElementwiseAngle2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements):
        phi = nn.Sequential(
            m_nn.ElementwiseLinear(in_channel=gnn_qubits),
            m_nn.QuantumMLP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements),
            )
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)

class QuantumIQP2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements):
        phi = m_nn.QuantumSphericalIQP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements)
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)

class QuantumElementwiseIQP2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements):
        phi = nn.Sequential(
            m_nn.ElementwiseLinear(in_channel=gnn_qubits),
            m_nn.QuantumSphericalIQP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements),
            )
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)

# %%
def train(model, data_module, train_info):
    # setup wandb logger
    if cf["wandb"]:
        wandb_info = {}
        wandb_info["project"]  = cf["project"]
        wandb_info["group"]    = f"{train_info['sig']}_{train_info['bkg']}"
        wandb_info["name"]     = f"{train_info['group_rnd']} | {cf['time']}_{train_info['rnd_seed']}"
        wandb_info["id"]       = wandb_info["name"]
        wandb_info["save_dir"] = root_dir 
        wandb_logger = WandbLogger(**wandb_info)
        wandb_config = {}
        wandb_config.update(cf)
        wandb_config.update(train_info)
        wandb_config.update(wandb_info)
        wandb_logger.experiment.config.update(wandb_config)
        wandb_logger.watch(model, log="all")

    # start lightning training
    logger  = wandb_logger if cf["wandb"] else None
    trainer = L.Trainer(
        logger            = logger, 
        accelerator       = cf["accelerator"],
        max_epochs        = cf["max_epochs"],
        fast_dev_run      = cf["fast_dev_run"],
        log_every_n_steps = cf["log_every_n_steps"],
        )
    litmodel = m_lightning.BinaryLitModel(model, graph=True)
    trainer.fit(litmodel, datamodule=data_module)
    trainer.test(litmodel, datamodule=data_module)

    # finish wandb monitoring
    if cf["wandb"]:
        wandb.finish()

# %%
data_abbrev = {"ZprimeToZhToZinvhbb":"Zinvhbb", "ZprimeToZhToZlephbb":"Zlephbb", "QCD_HT1500to2000":"QCD2000", "QCD_HT2000toInf":"QCDinf"}

for R, num_bin_data in list(product([1, 0.5, 0.1], [7500, 5000, 2500, 1000, 500])):
    data_info_list = [
        {"sig":"ZprimeToZhToZinvhbb", "bkg":"QCD_HT2000toInf", "jet_type":"fatjet", "subjet_radius":R ,"cut_limit":None, "bin":1, "num_bin_data":num_bin_data},
        {"sig":"ZprimeToZhToZinvhbb", "bkg":"QCD_HT1500to2000", "jet_type":"fatjet", "subjet_radius":R ,"cut_limit":None, "bin":1, "num_bin_data":num_bin_data},
        {"sig":"ZprimeToZhToZlephbb", "bkg":"QCD_HT2000toInf", "jet_type":"fatjet", "subjet_radius":R ,"cut_limit":None, "bin":1, "num_bin_data":num_bin_data},
        {"sig":"ZprimeToZhToZlephbb", "bkg":"QCD_HT1500to2000", "jet_type":"fatjet", "subjet_radius":R ,"cut_limit":None, "bin":1, "num_bin_data":num_bin_data},
    ]
    for data_info in data_info_list:
        sig_buffer = d_hep_data.UniformBinJetBuffer(channel=data_info["sig"], num_events=50000, **data_info)
        bkg_buffer = d_hep_data.UniformBinJetBuffer(channel=data_info["bkg"], num_events=50000, **data_info)
        for rnd_seed in range(3):
            L.seed_everything(rnd_seed)
            sig_events  = sig_buffer.get_uniform_bin_data()
            bkg_events  = bkg_buffer.get_uniform_bin_data()
            data_suffix = f"{data_abbrev[data_info['sig']]}_{data_abbrev[data_info['bkg']]}_cut{data_info['cut_limit']}_bin{data_info['bin']}-{data_info['num_bin_data']}_R{data_info['subjet_radius']}"

            # classical
            preprocess_mode = ""
            gnn_in, gnn_out, gnn_hidden, gnn_layers = 6, 6, 0, 0
            model = Classical2PCGNN(gnn_in=gnn_in, gnn_out=gnn_out, gnn_hidden=gnn_hidden, gnn_layers=gnn_layers)
            data_module = JetDataModule(sig_events, bkg_events, preprocess_mode)
            train_info = {"rnd_seed":rnd_seed, "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            train_info["group_rnd"] = f"{model.__class__.__name__}_{preprocess_mode}_go{gnn_out}_gh{gnn_hidden}_gl{gnn_layers} | {data_suffix}"
            train_info.update(data_info)
            train(model, data_module, train_info)

            # classical with normalized data
            # preprocess_mode = "normalize"
            # gnn_in, gnn_out, gnn_hidden, gnn_layers = 6, 6, 0, 0
            # model = Classical2PCGNN(gnn_in=gnn_in, gnn_out=gnn_out, gnn_hidden=gnn_hidden, gnn_layers=gnn_layers, mlp_hidden=mh, mlp_layers=ml)
            # data_module = JetDataModule(sig_events, bkg_events, preprocess_mode)
            # train_info = {"rnd_seed":rnd_seed, "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            # train_info["group_rnd"]  = f"{model.__class__.__name__}_{preprocess_mode}_go{gnn_out}_gh{gnn_hidden}_gl{gnn_layers} | {data_suffix}"
            # train_info.update(data_info)
            # train(model, data_module, train_info)

            # # quantum angle encoding
            # preprocess_mode = "normalize"
            # gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements = 6, 2, 0, [[i, "Z"] for i in range(6)]
            # model = QuantumAngle2PCGNN(gnn_qubits=gnn_qubits, gnn_layers=gnn_layers, gnn_reupload=gnn_reupload, gnn_measurements=gnn_measurements)
            # data_module = JetDataModule(sig_events, bkg_events, preprocess_mode)
            # train_info = {"rnd_seed":rnd_seed, "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            # train_info["group_rnd"]  = f"{model.__class__.__name__}_{preprocess_mode}_q{gnn_qubits}_gl{gnn_layers}_gr{gnn_reupload} | {data_suffix}"
            # train_info.update(data_info)
            # # train(model, data_module, train_info)

            # # quantum angle encoding with elementwise linear
            # preprocess_mode = "normalize"
            # gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements = 6, 2, 0, [[i, "Z"] for i in range(6)]
            # model = QuantumElementwiseAngle2PCGNN(gnn_qubits=gnn_qubits, gnn_layers=gnn_layers, gnn_reupload=gnn_reupload, gnn_measurements=gnn_measurements)
            # data_module = JetDataModule(sig_events, bkg_events, preprocess_mode)
            # train_info = {"rnd_seed":rnd_seed, "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            # train_info["group_rnd"]  = f"{model.__class__.__name__}_{preprocess_mode}_q{gnn_qubits}_gl{gnn_layers}_gr{gnn_reupload} | {data_suffix}"
            # train_info.update(data_info)
            # # train(model, data_module, train_info)

            # # quantum IQP encoding
            # preprocess_mode = "normalize"
            # gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements = 6, 2, 0, [[i, "Z"] for i in range(6)]
            # model = QuantumIQP2PCGNN(gnn_qubits=gnn_qubits, gnn_layers=gnn_layers, gnn_reupload=gnn_reupload, gnn_measurements=gnn_measurements)
            # data_module = JetDataModule(sig_events, bkg_events, preprocess_mode)
            # train_info = {"rnd_seed":rnd_seed, "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            # train_info["group_rnd"]  = f"{model.__class__.__name__}_{preprocess_mode}_q{gnn_qubits}_gl{gnn_layers}_gr{gnn_reupload} | {data_suffix}"
            # train_info.update(data_info)
            # # train(model, data_module, train_info)

            # # quantum IQP encoding with elementwise linear
            # preprocess_mode = "normalize"
            # gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements = 6, 2, 0, [[i, "Z"] for i in range(6)]
            # model = QuantumElementwiseIQP2PCGNN(gnn_qubits=gnn_qubits, gnn_layers=gnn_layers, gnn_reupload=gnn_reupload, gnn_measurements=gnn_measurements)
            # data_module = JetDataModule(sig_events, bkg_events, preprocess_mode)
            # train_info = {"rnd_seed":rnd_seed, "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            # train_info["group_rnd"]  = f"{model.__class__.__name__}_{preprocess_mode}_q{gnn_qubits}_gl{gnn_layers}_gr{gnn_reupload} | {data_suffix}"
            # train_info.update(data_info)
            # # train(model, data_module, train_info)