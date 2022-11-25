import numpy as np
import uproot
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
    def __init__(self, channel, num_events, jet_type, cut=None, expressions=None):
        '''
            - jet_type: "jet" or "fatjet"
            - keep_by: keep criterion, by "e" or "pt" or "mass", 
                default use "pt" -> high quality in barrel
        '''
        # read channel root file
        dir_path  = "/Users/yianchen/CMS_Open_Data_Workspace/CMSSW_7_6_7/src/QCD_Jet_Fatjet/Analyzer/root_files"
        root_path = f"{dir_path}/{channel}_{num_events}.root"
        events = uproot.open(root_path + ":jets/Events")
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

if __name__ == '__main__':
    channel, num_events, jet_type = "ZprimeToZhToZlephbb", 100, "fatjet"
    cut = f"({jet_type}_pt >= 800) & ({jet_type}_pt <= 1200)"
    expressions = ["fatjet_daughter_pt", "fatjet_daughter_eta"]
    jet_events = JetEvents(channel, num_events, jet_type, cut, expressions)
    print(len(jet_events.events))
    print(jet_events.events["fatjet_pt"])
    print(jet_events.events["fatjet_daughter_pt"])