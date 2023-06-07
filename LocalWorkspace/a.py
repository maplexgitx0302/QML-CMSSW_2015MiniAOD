# %%
"""
## Setup
"""

# %%
# basic packages
import os, time, random
from itertools import product
import matplotlib.pyplot as plt

# module packages
import m_nn
import m_lightning

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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import torch_geometric.nn as geom_nn

# wandb
import wandb
from lightning.pytorch.loggers import WandbLogger

# wandb login
wandb.login()

# reproducibility
L.seed_everything(3020616)

# faster calculation on GPU but less precision
torch.set_float32_matmul_precision("medium")

# %%
# configuration dictionary
cf = {}
cf["time"]     = time.strftime("%Y%m%d_%H%M%S", time.localtime())
cf["wandb"]    = True
cf["project"]  = "g_polyhedron_geognn"
cf["rnd_seed"] = None # to be determined by for loop

# data
cf['sig_channel'] = "tetrahedron"
cf['bkg_channel'] = "cube"
cf['num_edges']   = 1
cf["commit"]      = "center" # set center of polyhedron -> center of pos = (0,0,0)

# model
cf['gnn_layers'] = 2

# traning configuration
cf["learning_rate"]     = 1E-3
cf["num_data"]          = 500
cf["batch_size"]        = 64
cf["num_workers"]       = 0
cf["max_epochs"]        = 100
cf["accelerator"]       = "cpu"
cf["log_every_n_steps"] = cf["batch_size"] // 2

# %%
"""
## Data Module
"""

# %%
class PolyhedronDataModule(pl.LightningDataModule):
    def __init__(self, num_data, num_edges, coordinate):
        super().__init__()
        data_list_1 = self._create_data("tetrahedron", num_data, num_edges, coordinate, y=1)
        data_list_0 = self._create_data("cube", num_data, num_edges, coordinate, y=0)
        random.shuffle(data_list_1)
        random.shuffle(data_list_0)
        ratio       = 0.8
        num_train   = int(ratio * num_data)
        self.train_dataset = data_list_1[:num_train] + data_list_0[:num_train]
        self.test_dataset  = data_list_1[num_train:] + data_list_0[num_train:]

    def _generate_random_unit_vector(self):
        # for the algorithm, see
        # 1. https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
        # 2. https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        while True:
            gaussian_vector  = np.random.randn(3)
            vector_magnitude = np.linalg.norm(gaussian_vector)
            if vector_magnitude > 1e-5:
                break
        random_unit_vector = gaussian_vector / vector_magnitude
        return random_unit_vector

    def _rotation_matrix(self, unit_vector, theta):
        # see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        s = np.sin(theta)
        c = np.cos(theta)
        x, y, z = unit_vector
        return np.array([
            [c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
            [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
            [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]
        ])

    def _randomly_rotate(self, nodes):
        # the primary node will always be the [0] node
        primary_node       = nodes[0]
        # randomly choose a unit vector
        random_unit_vector = self._generate_random_unit_vector()
        # get the unit vector perpendicular to the primary node and the random unit vector
        perp_unit_vector   = np.cross(primary_node, random_unit_vector)
        perp_unit_vector   = perp_unit_vector / np.linalg.norm(perp_unit_vector)
        # get the angle between the primary node and the random unit vector
        p, n, v = primary_node, random_unit_vector, perp_unit_vector
        if np.sign(np.dot(p, n)) > 0:
            angle = np.arcsin(min(np.linalg.norm(perp_unit_vector), 1))
        else:
            angle = np.pi-np.arcsin(min(np.linalg.norm(perp_unit_vector), 1))
        # rotate primary nodes and others with angle
        nodes = nodes.T
        nodes = self._rotation_matrix(v, angle) @ nodes
        # rotate an arbitrary angle with axis n
        theta = 2 * np.pi * np.random.rand()
        nodes = self._rotation_matrix(n, theta) @ nodes
        nodes = nodes.T
        return nodes

    def _polyhedron_nodes(self, polyhedron):
        # tetrahedron
        if polyhedron == "tetrahedron":
            return (1/np.sqrt(3)) * np.array([
                [1,1,1], [-1,-1,1], [-1,1,-1], [1,-1,-1]
            ])
        # cube
        elif polyhedron == "cube":
            return (1/np.sqrt(3)) * np.array([
                [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
                [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
            ])

    def _create_data(self, polyhedron, num_data, num_edges, coordinate, y):
        data_list = []
        for _ in range(num_data):
            # get nodes coordinates
            nodes = self._polyhedron_nodes(polyhedron)
            # first rotating primary nodes to a random vector then random rotate around it
            nodes = self._randomly_rotate(nodes)
            # randomly add a small noise
            nodes = nodes + 1E-3 * np.random.rand(*np.shape(nodes))
            # adding edges, with each node has num_edges edges
            edges = []
            for i in range(len(nodes)):
                # distance between two nodes
                distances = [np.linalg.norm(nodes[i] - nodes[j]) for j in range(len(nodes))]
                # the first arg is trivial self distance
                edges += [(i, idx) for idx in np.argsort(distances)[1:num_edges+1]]
            edges = torch.tensor(edges).transpose(0, 1)
            # select Cartesian coordinates or Spherical coordinates
            nodes = torch.tensor(nodes, dtype=torch.float32)
            if coordinate == "cartesian":
                pass
            elif coordinate == "spherical":
                nodes_r     = torch.sqrt(torch.sum(nodes**2, dim=1))
                nodes_theta = torch.acos(nodes[:,2] / nodes_r).reshape(-1, 1)
                nodes_phi   = torch.atan2(nodes[:,1], nodes[:,0]).reshape(-1, 1)
                nodes = torch.cat([nodes_theta, nodes_phi], dim=1)
            else:
                raise ValueError(f"Unknown coordinate type: {coordinate}")
            # check whether nodes are valid -> no nan values
            if torch.any(torch.isnan(nodes)):
                raise ValueError(f"{nodes}")
            data_list.append(Data(x=nodes, edge_index=edges, y=y))
        return data_list

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=cf["batch_size"], num_workers=cf["num_workers"], shuffle=True)
    def val_dataloader(self):
        # choose val data to be same as test data, since we just want to monitoring the behavior
        return DataLoader(self.test_dataset, batch_size=cf["batch_size"], num_workers=cf["num_workers"])
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=cf["batch_size"], num_workers=cf["num_workers"])

# %%
"""
## Model
"""

# %%
class GeoMessagePassing(MessagePassing):
    def __init__(self, phi, gamma, aggr):
        super().__init__(aggr=aggr, flow="target_to_source")
        self.phi   = phi
        self.gamma = gamma
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.phi(torch.cat((x_i, x_j), dim=-1))
    def update(self, aggr_out, x):
        return self.gamma(torch.cat((x, aggr_out), dim=-1))

class ClassicalGeoGNN(nn.Module):
    def __init__(self, num_features, gnn_layers):
        super().__init__()
        gnn_phi   = m_nn.ClassicalMLP(in_channel=2*num_features, out_channel=num_features, hidden_channel=2*num_features, num_layers=gnn_layers)
        gnn_gamma = m_nn.ClassicalMLP(in_channel=2*num_features, out_channel=1, hidden_channel=0, num_layers=0)
        self.gnn  = GeoMessagePassing(gnn_phi, gnn_gamma, aggr="sum")
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch)
        return x

class QuantumGeoGNN(nn.Module):
    def __init__(self, num_features, gnn_layers):
        super().__init__()
        gnn_meas  = [[i, "X"] for i in range(num_features)]
        gnn_phi   = m_nn.QuantumMLP(num_qubits=2*num_features, num_layers=gnn_layers, num_reupload=1, measurements=gnn_meas)
        gnn_gamma = m_nn.ClassicalMLP(in_channel=2*num_features, out_channel=1, hidden_channel=0, num_layers=0)
        self.gnn  = GeoMessagePassing(gnn_phi, gnn_gamma, aggr="sum")
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch)
        return x

# %%
"""
## Training
"""

# %%
def train(model, data_module, commit="", suffix=""):
    # wandb logger setup
    if cf["wandb"]:
        # setup id and path for saving
        project  = cf['project']
        group    = f"{cf['time']}_{commit}_{cf['sig_channel']}_{cf['bkg_channel']}"
        job_type = f"{cf['sig_channel']}_{cf['bkg_channel']}_{cf['num_edges']}"
        name     = f"{model.__class__.__name__}_gl{cf['gnn_layers']}_{suffix} | {job_type} | rnd_{cf['rnd_seed']} | {cf['time']}"
        id       = f"{name}"
        # additional information
        cf["model_structure"] = f"gl{cf['gnn_layers']}"
        cf["model_name"]      = model.__class__.__name__
        cf["group_rnd_seed"]  = f"{cf['model_name']}_{cf['model_structure']}_{suffix} | {job_type}"
        cf["suffix"]          = suffix
        # tags
        tags = [model.__class__.__name__] + [str(cf[key]) for key in cf.keys() if cf[key] is not None]
        # wandb logger setup
        wandb_logger = WandbLogger(project=project, group=group, job_type=job_type, name=name, id=id, save_dir=f"./result", tags=tags)
        wandb_logger.experiment.config.update(cf)
        wandb_logger.watch(model, log="all")

    # start lightning training
    logger   = wandb_logger if cf["wandb"] else None
    trainer  = L.Trainer(
        logger=logger, 
        accelerator       = cf["accelerator"],
        max_epochs        = cf["max_epochs"],
        log_every_n_steps = cf["log_every_n_steps"],
        )
    # LightningModule
    litmodel = m_lightning.BinaryLitModel(model, lr=cf["learning_rate"], graph=True)
    trainer.fit(litmodel, datamodule=data_module)
    trainer.test(litmodel, datamodule=data_module)

    # finish wandb monitoring
    if cf["wandb"]:
        wandb.finish()

# %%
for edge in range(3):
    cf["num_edges"] = edge + 1
    for rnd_seed in range(3):
        # setup
        cf["rnd_seed"] = rnd_seed
        L.seed_everything(cf["rnd_seed"])

        # data module
        data_cartesian = PolyhedronDataModule(num_data=cf["num_data"], num_edges=cf["num_edges"], coordinate="cartesian")
        data_spherical = PolyhedronDataModule(num_data=cf["num_data"], num_edges=cf["num_edges"], coordinate="spherical")

        # cartesian
        cartesian_2pcnn  = ClassicalGeoGNN(num_features=3, gnn_layers=cf["gnn_layers"])
        cartesian_2pcqnn = QuantumGeoGNN(num_features=3, gnn_layers=cf["gnn_layers"])
        train(cartesian_2pcnn, data_cartesian, commit=cf["commit"], suffix="cartesian")
        train(cartesian_2pcqnn, data_cartesian, commit=cf["commit"], suffix="cartesian")

        #spherical
        spherical_2pcnn  = ClassicalGeoGNN(num_features=2, gnn_layers=cf["gnn_layers"])
        spherical_2pcqnn = QuantumGeoGNN(num_features=2, gnn_layers=cf["gnn_layers"])
        train(spherical_2pcnn, data_spherical, commit=cf["commit"], suffix="spherical")
        train(spherical_2pcqnn, data_spherical, commit=cf["commit"], suffix="spherical")