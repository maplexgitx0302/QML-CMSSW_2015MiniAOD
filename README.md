# QML w/ CMSSW Open Data (Run 2)

This project tests the performance of Quantum Machine Learning on High Energy Physics data. We use [CMS open data (miniaodsim format)](https://opendata.cern.ch/search?page=1&size=20&experiment=CMS&year=2015&file_type=miniaodsim), and extract the data into root files with [POET-2015MiniAOD](https://github.com/cms-opendata-analyses/PhysObjectExtractorTool/tree/2015MiniAOD), following the guide from [CMS Open Data Workshop 2022](https://cms-opendata-workshop.github.io/2022-08-01-cms-open-data-workshop/).

### Prerequisites
1. A docker environment (optional): See [CMSSW docker tutorial](https://cms-opendata-workshop.github.io/workshop2022-lesson-docker/)
    - Example data buffer are already in `LocalWorkspace/data_buffer`, if you don't nedd to test new data, ignore docker environment.

2. Python packages (may simply install with `pip install -r requirements.txt`):
   - HEP open data 
     - [awkward](https://awkward-array.org/quickstart.html): Data structure for jagged data.
     - [uproot](https://uproot.readthedocs.io/en/latest/): Read `.root` files with python and represent in `awkward.Array` format.
   - Machine Learning
     - [pytorch](https://pytorch.org): Classical package for machine learning
     - [pytorch-lightning](https://www.pytorchlightning.ai): Powerful tool to make pytorch code more simplier and clear.
     - [wandb](https://docs.wandb.ai): Monitoring weights and biases, logging loss and accuracy or other metrics.
     - [pennylane](https://pennylane.ai): Quantum machine learning.
   - Analysis
     - [qympy](https://github.com/r08222011/Qympy): For symbolic calculation of quantum circuits.

### Instructions
Two main workspaces:
1. LocalWorkspace: To be run in your local.
   - HEP open data
     - `hep_events.py`: Read `.root` files and turn into `awkward.Array` format.
     - `hep_buffer.py`: Extract only specific features and save as `torch.Tensor` format.
   - Machine Learning
     - `pl_arckernel.ipynb`: Main code for training.
2. QCD_Jet_Fatjet: To be run in the docker (otherwise need `CMSSW` in you local).
    - `plugins/Analyzer.cc`: Analyzer for extracting hep data.
    - `python/ConfFile_cfg.py`: To be run with command `cmsRun`.
    - `root_files/download.py`: Can efficiently download with many channels.