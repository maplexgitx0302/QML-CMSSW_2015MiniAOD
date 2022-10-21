import numpy as np
import awkward as ak

def max_selection(events, prefix, key, return_cut=False):
    '''
        events : events from uproot
        prefix : "jet" or "fatjet"
        key    : "e", "pt", "eta", "phi", "ch", "mass"
    '''
    key_array = events[f"{prefix}_{key}"].array()
    key_cut   = ak.firsts(ak.argsort(key_array, ascending=False))
    key_cut   = ak.unflatten(key_cut, counts=np.ones(len(key_array), dtype=int))
    features  = ["e", "pt", "eta", "phi", "ch", "mass"]
    daughter  = {}
    for feature in features:
        feature_array = events[f"{prefix}_daughter_{feature}"].array()
        feature_array = feature_array[key_cut]
        feature_array = ak.flatten(feature_array, axis=1)
        daughter[feature] = feature_array
    if return_cut: return daughter, key_cut
    else: return daughter