import os
import subprocess

num_events = 10000
channels = [
    "ZprimeToZhToZinvhbb",
    "ZprimeToZhToZlephbb",
    "QCD_HT1500to2000",
    "QCD_HT2000toInf",
]

for channel in channels:
    main_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.dirname(main_path) + "/python/Remote_ConfFile_cfg.py"

    f = open(main_path+"/config.txt", "w")
    f.writelines([channel, " ", str(num_events)])
    f.close()

    subprocess.call(["cmsRun", config_path])