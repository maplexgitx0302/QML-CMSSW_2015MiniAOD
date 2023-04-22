# %%
"""
### Packages
"""

# %%
# basic
import os, time, itertools

# qml
import pennylane as qml
from pennylane import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# pytorch_lightning
import lightning as L
import lightning.pytorch as pl

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

# current time
global_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# %%
"""
### QML NN Model
"""

# %%
class MLP(nn.Module):
    def __init__(self, q_device, q_diff, q_interface, num_qubits, num_reupload, num_qlayers, num_clayers, num_chidden):
        super().__init__()
    
        # create a quantum MLP
        @qml.qnode(qml.device(q_device, wires=num_qubits), diff_method=q_diff, interface=q_interface)
        def circuit(inputs, weights):
            for i in range(num_reupload):
                qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights=weights[i], wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        
        # turn the quantum circuit into a torch layer
        weight_shapes = {"weights":(num_reupload, num_qlayers, num_qubits, 3)}
        q_net = [qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)]

        # classical mlp
        c_in, c_hidden, c_out = num_qubits, num_chidden, 1
        c_net = [nn.Linear(c_in, c_hidden), nn.ReLU()]
        for _ in range(num_clayers-2):
            c_net += [nn.Linear(c_hidden, c_hidden), nn.ReLU()]
        c_net += [nn.Linear(c_hidden, c_out)]

        # combine classical and quantum net
        self.net = nn.Sequential(*(q_net + c_net))

    def forward(self, x):
        return self.net(x)

# %%
"""
### Lit Model
"""

# %%
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, data):
        # predict y
        x, y_true = data
        x = self.model(x)
        x = x.squeeze(dim=-1)

        # calculate loss and accuracy
        y_pred = x > 0
        loss   = self.loss_function(x, y_true.float())
        acc    = (y_pred == y_true).float().mean()
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1E-3)
        return optimizer
    
    def on_train_epoch_start(self):
        self.start_time = time.time()
    def on_train_epoch_end(self):
        self.end_time = time.time()
        delta_time = self.end_time - self.start_time
        self.log("epoch_time", delta_time, on_step=False, on_epoch=True)

    def training_step(self, data, batch_idx):
        loss, acc = self.forward(data)
        # self.log("train_loss", loss, on_step=True, on_epoch=True)
        # self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

# %%
"""
### Dataset
"""

# %%
class RandomDataset(Dataset):
    def __init__(self, num_data, num_dim):
        super().__init__()
        self.x = torch.rand(num_data, num_dim)
        self.y = torch.cat((torch.ones(num_data//2), torch.zeros(num_data//2)))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class RandomDataModule(pl.LightningDataModule):
    def __init__(self, num_data, num_dim, batch_size, num_workers):
        super().__init__()
        self.train_dataset  = RandomDataset(num_data, num_dim)
        self.batch_size     = batch_size
        self.num_workers    = num_workers
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,  shuffle=True)

# %%
"""
### Training Procedure
"""

# %%
def test_time(wb, c_device, model_config, data_config, group_prefix="", group_suffix="", job_type_prefix="", job_type_suffix="", name_prefix="", name_suffix=""):
    model       = MLP(**model_config)
    litmodel    = LitModel(model)
    data_module = RandomDataModule(**data_config)

    mcf = model_config
    dcf = data_config
    if wb == True:
        # group
        group = ""
        if group_prefix != "":
            group += group_prefix + "_"
        group += f"dim{dcf['num_dim']}_rl{mcf['num_reupload']}_ql{mcf['num_qlayers']}_cl{mcf['num_clayers']}"
        if group_suffix != "":
            group += "_" + group_suffix

        # job_type
        job_type = ""
        if job_type_prefix != "":
            job_type += job_type_prefix + "_"
        job_type += f"{c_device}|{mcf['q_device']}|diff_{mcf['q_diff']}|interface_{mcf['q_interface']}"
        if job_type_suffix != "":
            job_type += "_" + job_type_suffix

        # name
        name  = global_time + "_"
        if name_prefix != "":
            name += name_prefix + "_"
        name  += f"{job_type}_batch{dcf['batch_size']}_worker{dcf['num_workers']}_dim{dcf['num_dim']}"
        if name_suffix != "":
            name += "_" + name_suffix

        # id
        id = group + "_" + job_type + "_" + name

        # wandb logger
        wandb_logger = WandbLogger(project="t_qml_time", group=group, job_type=job_type, name=name, id=id, save_dir=f"./result")
        wandb_logger.experiment.config.update(mcf)
        wandb_logger.experiment.config.update(dcf)
        wandb_logger.watch(model, log="all")
        logger = wandb_logger
    else:
        logger = None

    trainer  = L.Trainer(
        logger      = logger, 
        accelerator = c_device, 
        max_epochs  = 3,
        )

    trainer.fit(litmodel, datamodule=data_module)

    if wb:
        wandb.finish()

# %%
real_test = True

if real_test:
    wb = True
    l_num_dim     = [4, 8, 12, 16]
    l_cpu_batch   = [16, 32, 64]
    l_gpu_batch   = [16, 32, 64]
    l_num_workers = [0]
    l_product     = list(itertools.product(["cpu"], l_num_dim, l_cpu_batch, l_num_workers)) + list(itertools.product(["gpu"], l_num_dim, l_gpu_batch, l_num_workers))

# else:
#     wb = False
#     l_num_dim     = [16]
#     l_cpu_batch   = [256]
#     l_gpu_batch   = [64]
#     l_num_workers = [0]
#     l_product     = list(itertools.product(["cpu"], l_num_dim, l_cpu_batch, l_num_workers)) + list(itertools.product(["gpu"], l_num_dim, l_gpu_batch, l_num_workers))

q_tuple = [
    # (q_device, q_diff, q_interface)
    ("default.qubit", "best", "auto"),
    ("default.qubit", "best", "torch"),
    ("default.qubit", "backprop", "auto"),
    ("default.qubit", "backprop", "torch"),
    ("lightning.qubit", "adjoint", "auto"),
    ("lightning.qubit", "adjoint", "torch"),
    # ("lightning.gpu", "adjoint", "auto"),
    # ("lightning.gpu", "adjoint", "torch"),
]

for c_device, num_dim, batch_size, num_workers in l_product:
    for q_device, q_diff, q_interface in q_tuple:
        model_config = {
            "q_device"     : q_device,
            "q_diff"       : q_diff,
            "q_interface"  : q_interface,
            "num_qubits"   : num_dim,
            "num_reupload" : 2,
            "num_qlayers"  : 1,
            "num_clayers"  : 2,
            "num_chidden"  : 4 * num_dim,
        }
        data_config = {
            "num_data"    : 512,
            "num_dim"     : num_dim,
            "batch_size"  : batch_size,
            "num_workers" : num_workers,
        }
        test_time(wb, c_device, model_config, data_config, job_type_prefix="ntugpu")