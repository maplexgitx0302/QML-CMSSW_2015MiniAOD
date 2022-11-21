import numpy as np
import awkward as ak

class JetEvents:
    def __init__(self, events, jet_type, keep_by="pt"):
        '''
            jet_type  : "jet" or "fatjet"
            select_by : keep criterion, by "e" or "pt" or "mass", 
                        default use "pt" -> high quality in barrel

        '''
        self.events = events
        self.jet_type, self.keep_by = jet_type, keep_by
        self.num_events = len(events[f"{jet_type}_{keep_by}"].array())
        self.nsubjettiness = {}
        self.daughter = {} # can be viewed as a tree
        self.keep_index = None # index of which jet to keep for each event
        self._extract_daughter()
        self._extract_pdgid()

    def _extract_daughter(self):
        # get criterion array and get max key value index
        criterion_array = self.events[f"{self.jet_type}_{self.keep_by}"].array()
        keep_index = ak.firsts(ak.argsort(criterion_array, ascending=False))
        keep_index = ak.unflatten(keep_index, counts=np.ones(len(criterion_array), dtype=int))
        self.keep_index = keep_index
        # add branch to self.daughter
        features = ["e", "pt", "eta", "phi", "ch", "mass", "pdgid", "tau1", "tau2", "tau3"]
        for feature in features:
            if feature in ["e", "pt", "eta", "phi", "ch", "mass", "pdgid"]:
                feature_array = self.events[f"{self.jet_type}_daughter_{feature}"].array()
                feature_array = feature_array[keep_index]
                feature_array = ak.flatten(feature_array, axis=1)
                self.daughter[feature] = feature_array
            elif self.jet_type == "fatjet" and feature in ["tau1", "tau2", "tau3"]:
                feature_array = self.events[f"{self.jet_type}_{feature}"].array()
                feature_array = feature_array[keep_index]
                feature_array = ak.flatten(feature_array, axis=1)
                self.nsubjettiness[feature] = feature_array

    def _extract_pdgid(self):
        electron   = np.abs(self.daughter["pdgid"]) == 11 # electron
        muon       = np.abs(self.daughter["pdgid"]) == 13 # muon
        gamma      = np.abs(self.daughter["pdgid"]) == 22 # gamma
        ch_hadron  = np.abs(self.daughter["pdgid"]) == 211 # charged hadrons
        neu_hadron = np.abs(self.daughter["pdgid"]) == 130 # neutral hadrons
        HF_hadron  = np.abs(self.daughter["pdgid"]) == 1 # hadronic particles in HF
        HF_em      = np.abs(self.daughter["pdgid"]) == 2 # em particles in HF
        # check whether other types of particles exist
        accumulate_particles = 0
        accumulate_particles += ak.sum(electron) + ak.sum(muon) + ak.sum(gamma)
        accumulate_particles += ak.sum(ch_hadron) + ak.sum(neu_hadron)
        accumulate_particles += ak.sum(HF_hadron) + ak.sum(HF_em)
        total_particles = ak.sum(ak.ones_like(self.daughter["pdgid"]))
        if accumulate_particles != total_particles:
            pdgid_ratio = accumulate_particles / total_particles
            raise Exception(f"Other types of pdgid exists: pdgid_ratio = {pdgid_ratio}")
        # set boolean branches into tree
        self.daughter["electron"]   = electron
        self.daughter["muon"]       = muon
        self.daughter["gamma"]      = gamma
        self.daughter["ch_hadron"]  = ch_hadron
        self.daughter["neu_hadron"] = neu_hadron
        self.daughter["HF_hadron"]  = HF_hadron
        self.daughter["HF_em"]      = HF_em