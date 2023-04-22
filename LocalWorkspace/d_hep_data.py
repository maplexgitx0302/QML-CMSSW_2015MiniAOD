import os
import uproot
import awkward as ak
import fastjet
import numpy as np
import random

pdgid_table = {
    "electron": 11,
    "muon": 13,
    "gamma": 22,
    "ch_hadron": 211,
    "neu_hadron": 130,
    "HF_hadron": 1,
    "HF_em": 2,
}

# run fastjet 1 time then it won't show up cite reminder
_jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
_array   = ak.Array([{"px": 0.1, "py": 0.2, "pz": 0.3, "E": 0.4},])
_cluster = fastjet.ClusterSequence(_array, _jet_def)

class JetEvents:
    def __init__(self, channel:str, num_events:int, jet_type:str, cut:str=None):
        self.channel    = channel    # signal or background channel
        self.num_events = num_events # the "num_events" suffix of the loaded root file
        self.jet_type   = jet_type   # "jet" or "fatjet"
        self.cut        = cut        # threshold to cut off the events
        self.events     = self._read_root_file()

    def _read_root_file(self):
        # use "uproot" package to read the root file
        src_path  = os.path.expanduser("~/CMS_Open_Data_Workspace/CMSSW_7_6_7/src")
        dir_path  = f"{src_path}/QCD_Jet_Fatjet/Analyzer/root_files"
        root_path = f"{dir_path}/{self.channel}_{self.num_events}.root"
        events    = uproot.open(root_path + ":jets")
        if len(events.keys()) > 1:
            events = events[max(events.keys(), key=lambda x: x.split(";")[-1])]
        else:
            events = events["Events"]

        # select jet or fatjet and apply cuts, note different data types should be splited
        jt = self.jet_type
        int_expressions = [
                f"{jt}_ch", f"{jt}_daughter_ch", 
                f"{jt}_daughter_pdgid", 
            ]
        float_expressions = [
                f"{jt}_e", f"{jt}_mass", 
                f"{jt}_px", f"{jt}_py", f"{jt}_pz",
                f"{jt}_pt", f"{jt}_eta", f"{jt}_phi",
                f"{jt}_daughter_e", f"{jt}_daughter_mass",
                f"{jt}_daughter_px", f"{jt}_daughter_py", f"{jt}_daughter_pz",
                f"{jt}_daughter_pt", f"{jt}_daughter_eta", f"{jt}_daughter_phi",
            ]
        int_events   = events.arrays(expressions=int_expressions, cut=self.cut)
        float_events = events.arrays(expressions=float_expressions, cut=self.cut)
        events       = float_events
        for field in int_events.fields:
            events[field] = int_events[field]
        events = events[ak.num(events)[f"{jt}_pt"] > 0]

        # get the event with highest transverse momentum pt
        max_index = ak.firsts(ak.argsort(events[f"{jt}_pt"], ascending=False), axis=1)
        max_index = ak.unflatten(max_index, counts=ak.ones_like(max_index))
        events = events[max_index]
        for fields in events.fields:
            try:
                events[fields] = ak.flatten(events[fields], axis=1)
            except ValueError as e:
                print(f"{fields} cannot be flattened: {e}")
        print(f"DataLog: Successfully create {self.channel} with {len(events)} events.")
        return events

    def fastjet_events(self, R, algorithm=fastjet.antikt_algorithm):
        # start clustering particles into jets
        fastjet_list = []
        for event in self.events:
            four_momentums = ak.Array({
                "px":event[f"{self.jet_type}_daughter_px"], 
                "py":event[f"{self.jet_type}_daughter_py"], 
                "pz":event[f"{self.jet_type}_daughter_pz"],
                "E" :event[f"{self.jet_type}_daughter_e"],
                })
            jet_def = fastjet.JetDefinition(algorithm, float(R))
            cluster = fastjet.ClusterSequence(four_momentums, jet_def)
            fastjet_list.append(cluster.inclusive_jets())
            
        # the output will only be 4-momentum of newly clustered jets
        fastjet_array          = ak.Array(fastjet_list)
        fastjet_array["e"]     = fastjet_array["E"]
        fastjet_array["p"]     = (fastjet_array["px"]**2 + fastjet_array["py"]**2 + fastjet_array["pz"]**2) ** 0.5
        fastjet_array["pt"]    = (fastjet_array["px"]**2 + fastjet_array["py"]**2) ** 0.5
        fastjet_array["theta"] = np.arccos(fastjet_array["pz"]/fastjet_array["p"])
        fastjet_array["eta"]   = -np.log(np.tan((fastjet_array["theta"])/2))
        fastjet_array["phi"]   = np.arctan2(fastjet_array["py"], fastjet_array["px"])

        # calculating delta-eta and delta-phi
        fastjet_array["delta_eta"] = fastjet_array["eta"] - self.events[f"{self.jet_type}_eta"]
        fastjet_array["delta_phi"] = fastjet_array["phi"] - self.events[f"{self.jet_type}_phi"]
        print(f"DataLog: Finish reclustering {self.channel} with anti-kt algorithm.")
        return fastjet_array
    
def events_uniform_Pt_weight(channel:str, num_events:int, jet_type:str, subjet_radius:float, cut_limit:tuple, bin:int, num_bin_data:int):
    # set bin info
    cut_lower, cut_upper = cut_limit
    bin_width = (cut_upper - cut_lower) / bin
    events = None
    fastjet_events = None

    # loop all bin
    print(f"Datalog: start creating uniform pt weight events")
    for i in range(bin):
        cut = f"({jet_type}_pt>={cut_lower+i*bin_width})&({jet_type}_pt<{cut_lower+(i+1)*bin_width})"
        bin_jet_events = JetEvents(channel=channel, num_events=num_events, jet_type=jet_type, cut=cut)
        bin_events = bin_jet_events.events
        bin_fastjet_events = bin_jet_events.fastjet_events(R=subjet_radius)
        assert len(bin_events) >= num_bin_data, f"num of bin_events smaller then num_bin_data: {len(bin_events)} < {num_bin_data}"

        idx = list(range(len(bin_events)))
        random.shuffle(idx)
        bin_events = bin_events[idx[:num_bin_data]]
        bin_fastjet_events = bin_fastjet_events[idx[:num_bin_data]]

        print(f"Datalog: bin ({i+1}/{bin}) | cut = {cut}")
        events = ak.concatenate((events, bin_events), axis=0) if events is not None else bin_events
        fastjet_events = ak.concatenate((fastjet_events, bin_fastjet_events), axis=0) if fastjet_events is not None else bin_fastjet_events
        
    # combine all fields
    for field in events.fields:
        try:
            _, feature = field.split("_")
            events[feature] = events[field]
        except:
            continue
    for field in fastjet_events.fields:
        events["_"+field] = fastjet_events[field]

    # other features
    events["p"] = (events["px"]**2 + events["py"]**2 + events["pz"]**2) ** 0.5
    events["_p"] = (events["_px"]**2 + events["_py"]**2 + events["_pz"]**2) ** 0.5
    events["theta"] = np.arccos(events["pz"]/events["p"])
    return events

# example
if __name__ == '__main__':
    jet_type    = "fatjet"
    cut         = f"({jet_type}_pt>=500)&({jet_type}_pt<=1500)",
    sig_channel = "ZprimeToZhToZinvhbb"
    bkg_channel = "QCD_HT2000toInf"

    sig_events = JetEvents(channel=sig_channel, num_events=50000, jet_type=jet_type, cut=cut)
    bkg_events = JetEvents(channel=bkg_channel, num_events=50000, jet_type=jet_type, cut=cut)

    def print_events(events):
        for field in events.fields:
            print(f"{field}: {events[field][0]}")

    print_events(sig_events.events)
    print("-" * 100)
    print_events(bkg_events.events)