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

def get_events(channel, num_events, jet_type, cut=None, expressions=None):
    '''Read root files with uproot then turn awkward arrays into torch arrays
    - channel(str): channel(process) name
    - num_events(int): total events of target data
    - jet_type(str): jet or fatjet
    - cut(str): criterion for selecting(cutting) the root data
    - expressions(list): features to be saved
    '''

    # read channel root file
    src_path  = "/Users/yianchen/CMS_Open_Data_Workspace/CMSSW_7_6_7/src"
    dir_path  = f"{src_path}/QCD_Jet_Fatjet/Analyzer/root_files"
    root_path = f"{dir_path}/{channel}_{num_events}.root"
    events = uproot.open(root_path + ":jets")
    if len(events.keys()) > 1:
        events = events[max(events.keys(), key=lambda x: x.split(";")[-1])]
    else:
        events = events["Events"]
    print(f"Log(get_events): Successfully open {root_path}")

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
    print(f"Log(get_events): Successfully create {channel} with {num_events} events.")
    return events

if __name__ == '__main__':
    channel, num_events, jet_type = "ZprimeToZhToZlephbb", 100, "fatjet"
    cut = f"({jet_type}_pt >= 800) & ({jet_type}_pt <= 1200)"
    expressions = ["fatjet_daughter_pt", "fatjet_daughter_eta"]
    events = get_events(channel, num_events, jet_type, cut, expressions)
    print(f"Length of events = {len(events)}")
    print(f"pt of parent events[0] = {events['fatjet_pt'][0]}")
    print(f"pt of daughter events[0] = {events['fatjet_daughter_pt'][0]}")