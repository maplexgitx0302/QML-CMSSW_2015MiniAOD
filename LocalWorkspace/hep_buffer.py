import os
import torch
import awkward as ak
from tqdm import tqdm
from hep_events import get_events

def load_data_buffer(channel, get_method, *args):
    suffix = " ".join(map(str, args))
    buffer_file = f"data_buffer/{channel}-{get_method.__name__}-{suffix}.pt"
    if not os.path.exists(buffer_file):
        print(f"Log(load_data_buffer): {channel} buffer not found, create now ...")
        events = get_method(channel, *args)
        torch.save(events, buffer_file)
    else:
        events = torch.load(buffer_file)
        print(f"Log(load_data_buffer): {channel} buffer found, loading complete!")
    return events

def get_parent_info(channel, num_events, jet_type, cut=None):
    expressions = [f"{jet_type}_{feature}" for feature in ["pt", "eta", "phi"]]
    if jet_type == "fatjet":
        expressions += ["fatjet_tau1", "fatjet_tau2", "fatjet_tau3"]
    events = get_events(channel, num_events, jet_type, cut, expressions)
    trimmed_events = torch.cat((
        torch.tensor(events[f"{jet_type}_pt"])[:, None], 
        torch.tensor(events[f"{jet_type}_eta"])[:, None],
        torch.tensor(events[f"{jet_type}_phi"])[:, None]), dim=1)
    if jet_type == "fatjet":
        trimmed_events = torch.cat((
            trimmed_events,
            torch.tensor(events["fatjet_tau1"])[:, None],
            torch.tensor(events["fatjet_tau2"])[:, None],
            torch.tensor(events["fatjet_tau3"])[:, None]), dim=1)
    return trimmed_events

def get_daughter_info(channel, num_events, num_particles, jet_type, cut=None):
    expressions = [f"{jet_type}_daughter_{feature}" for feature in ["pt", "eta", "phi"]]
    events = get_events(channel, num_events, jet_type, cut, expressions)
    trimmed_events = torch.zeros((len(events), num_particles*3))
    idx_argsort = ak.argsort(events[f"{jet_type}_daughter_pt"], axis=-1, ascending=False)
    for i in tqdm(range(len(events)), desc=f"get_daughter_info : Channel {channel} with {num_events} events"):
        l = min(len(idx_argsort[i]), num_particles)
        pt, eta, phi = torch.zeros(num_particles), torch.zeros(num_particles), torch.zeros(num_particles)
        pt[:l]  = torch.tensor(events[f"{jet_type}_daughter_pt"][i][idx_argsort[i][:l]])
        eta[:l] = torch.tensor(events[f"{jet_type}_daughter_eta"][i][idx_argsort[i][:l]])
        phi[:l] = torch.tensor(events[f"{jet_type}_daughter_phi"][i][idx_argsort[i][:l]])
        trimmed_events[i] = torch.cat((pt, eta, phi))
    return trimmed_events