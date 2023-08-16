''' To extract events from root files with package "uproot" and "awkward".
- uproot: for reading and extracting data from root files.
- awkward: to save data in a jagged data structure
'''

import os
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

def get_events(channel:str, num_events:int, jet_type:str, cut:str=None, expressions:list[str]=None) -> ak.Array:
    '''Read root files with uproot
    - channel: channel(process) name
    - num_events: total events of target data
    - jet_type: jet or fatjet
    - cut: criterion for selecting(cutting) the root data
    - expressions: features to be saved
    '''

    # read channel root file
    src_path  = os.path.expanduser("~/CMS_Open_Data_Workspace/CMSSW_7_6_7/src")
    dir_path  = f"{src_path}/QCD_Jet_Fatjet/Analyzer/root_files"
    root_path = f"{dir_path}/{channel}_{num_events}.root"
    events = uproot.open(root_path + ":jets")
    if len(events.keys()) > 1:
        events = events[max(events.keys(), key=lambda x: x.split(";")[-1])]
    else:
        events = events["Events"]
    print(f"DataLog(hep_events.py:get_events): Successfully open {root_path}")

    # select jet or fatjet and apply cuts
    if expressions == None:
        expressions = [
            # int format should be in front of float (without any reason ... to be clarified)
            f"{jet_type}_daughter_pdgid", f"{jet_type}_daughter_ch", f"{jet_type}_ch",
            # float
            f"{jet_type}_e", f"{jet_type}_pt", f"{jet_type}_eta", f"{jet_type}_phi", f"{jet_type}_mass", 
            f"{jet_type}_daughter_e", f"{jet_type}_daughter_pt", f"{jet_type}_daughter_eta", f"{jet_type}_daughter_phi",
            f"{jet_type}_daughter_mass",
            ]
        if jet_type == "fatjet":
            expressions += ["fatjet_tau1", "fatjet_tau2", "fatjet_tau3", "fatjet_tau2/fatjet_tau1", "fatjet_tau3/fatjet_tau2"]
        events = events.arrays(expressions=expressions, cut=cut)
    else:
        if f"{jet_type}_pt" not in expressions: expressions.append(f"{jet_type}_pt")
        events = events.arrays(expressions=expressions, cut=cut)
    events = events[ak.num(events)[f"{jet_type}_pt"] > 0]

    # get the highest pt jet from each events
    max_index = ak.firsts(ak.argsort(events[f"{jet_type}_pt"], ascending=False), axis=1)
    max_index = ak.unflatten(max_index, counts=ak.ones_like(max_index))
    events = events[max_index]
    for fields in events.fields:
        try:
            events[fields] = ak.flatten(events[fields], axis=1)
        except ValueError as e:
            print(f"{fields} cannot be flattened: {e}")

    print(f"DataLog(get_events): Successfully create {channel} with {num_events} events.")
    return events

# example
if __name__ == '__main__':
    # config
    channel     = "ZprimeToZhToZinvhbb"
    num_events  =  50000
    jet_type    = "jet"
    cut         = f"({jet_type}_pt >= 800) & ({jet_type}_pt <= 1200)"
    expressions = ["jet_daughter_pt", "jet_daughter_eta"]

    # extract data
    events = get_events(channel, num_events, jet_type, cut, expressions)
    print(f"Length of events = {len(events)}")
    print(f"pt of parent events[0] = {events['jet_pt'][0]}")
    print(f"pt of daughter events[0] = {events['jet_daughter_pt'][0]}")