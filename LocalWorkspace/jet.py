import numpy as np
import uproot
import awkward as ak
import torch
from tqdm import tqdm

pdgid_table = {
    "electron": 11,
    "muon": 13,
    "gamma": 22,
    "ch_hadron": 211,
    "neu_hadron": 130,
    "HF_hadron": 1,
    "HF_em": 2,
}

class JetEvents:
    def __init__(self, channel, num_events, jet_type, cut=None, expressions=None):
        # read channel root file
        src_path  = "/Users/yianchen/CMS_Open_Data_Workspace/CMSSW_7_6_7/src"
        dir_path  = f"{src_path}/QCD_Jet_Fatjet/Analyzer/root_files"
        root_path = f"{dir_path}/{channel}_{num_events}.root"
        events = uproot.open(root_path + ":jets")
        print(f"JetEvents : Successfully open {root_path}")
        if len(events.keys()) > 1:
            events = events[max(events.keys(), key=lambda x: x.split(";")[-1])]
        else:
            events = events["Events"]
        # select jet or fatjet and apply cuts
        if expressions == None:
            events = events.arrays(filter_name=f"{jet_type}*", cut=cut)
        else:
            if f"{jet_type}_pt" not in expressions: expressions.append(f"{jet_type}_pt")
            events = events.arrays(expressions=expressions, cut=cut)
        if cut !=  None:
            events = events[ak.num(events)[f"{jet_type}_pt"] > 0]
        max_index = ak.firsts(ak.argsort(events[f"{jet_type}_pt"], ascending=False))
        max_index = ak.unflatten(max_index, counts=ak.ones_like(max_index))
        events = events[max_index]
        for fields in events.fields:
            try:
                events[fields] = ak.flatten(events[fields], axis=1)
            except ValueError as e:
                print(f"{fields} cannot be flattened: {e}")
        self.events = events
        print(f"JetEvents : Successfully create {channel} with {num_events} events.")

def get_parent_info(channel, num_events, jet_type, cut=None):
    expressions = [f"{jet_type}_{feature}" for feature in ["pt", "eta", "phi"]]
    if jet_type == "fatjet":
        expressions += ["fatjet_tau1", "fatjet_tau2", "fatjet_tau3"]
    jet_events = JetEvents(channel, num_events, jet_type, cut, expressions)
    trimmed_events = torch.cat((
        torch.tensor(jet_events.events[f"{jet_type}_pt"])[:, None], 
        torch.tensor(jet_events.events[f"{jet_type}_eta"])[:, None],
        torch.tensor(jet_events.events[f"{jet_type}_phi"])[:, None]), dim=1)
    if jet_type == "fatjet":
        trimmed_events = torch.cat((
            trimmed_events,
            torch.tensor(jet_events.events["fatjet_tau1"])[:, None],
            torch.tensor(jet_events.events["fatjet_tau2"])[:, None],
            torch.tensor(jet_events.events["fatjet_tau3"])[:, None]), dim=1)
    return trimmed_events

def get_daughter_info(channel, num_events, num_particles, jet_type, cut=None):
    expressions = [f"{jet_type}_daughter_{feature}" for feature in ["pt", "eta", "phi"]]
    jet_events = JetEvents(channel, num_events, jet_type, cut, expressions)
    trimmed_events = torch.zeros((len(jet_events.events), num_particles*3))
    idx_argsort = ak.argsort(jet_events.events[f"{jet_type}_daughter_pt"], axis=-1, ascending=False)
    for i in tqdm(range(len(jet_events.events)), desc=f"get_daughter_info : Channel {channel} with {num_events} events"):
        l = min(len(idx_argsort[i]), num_particles)
        pt, eta, phi = torch.zeros(num_particles), torch.zeros(num_particles), torch.zeros(num_particles)
        pt[:l]  = torch.tensor(jet_events.events[f"{jet_type}_daughter_pt"][i][idx_argsort[i][:l]])
        eta[:l] = torch.tensor(jet_events.events[f"{jet_type}_daughter_eta"][i][idx_argsort[i][:l]])
        phi[:l] = torch.tensor(jet_events.events[f"{jet_type}_daughter_phi"][i][idx_argsort[i][:l]])
        trimmed_events[i] = torch.cat((pt, eta, phi))
    return trimmed_events

if __name__ == '__main__':
    channel, num_events, jet_type = "ZprimeToZhToZlephbb", 100, "fatjet"
    cut = f"({jet_type}_pt >= 800) & ({jet_type}_pt <= 1200)"
    expressions = ["fatjet_daughter_pt", "fatjet_daughter_eta"]
    jet_events = JetEvents(channel, num_events, jet_type, cut, expressions)
    print(len(jet_events.events))
    print(jet_events.events["fatjet_pt"])
    print(jet_events.events["fatjet_daughter_pt"])