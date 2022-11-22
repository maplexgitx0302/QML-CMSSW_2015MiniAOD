import numpy as np
import awkward as ak

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
    def __init__(self, events, jet_type, keep_by="pt"):
        '''
            - jet_type: "jet" or "fatjet"
            - keep_by: keep criterion, by "e" or "pt" or "mass", 
                default use "pt" -> high quality in barrel
        '''
        self.jet_type, self.keep_by = jet_type, keep_by
        events = events.arrays(filter_name=f"{jet_type}*")
        max_index = ak.firsts(ak.argsort(events[f"{self.jet_type}_{self.keep_by}"], ascending=False))
        max_index = ak.unflatten(max_index, counts=ak.ones_like(max_index))
        self.events = events[max_index]