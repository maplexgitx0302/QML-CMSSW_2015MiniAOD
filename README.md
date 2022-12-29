# QML w/ CMSSW Open Data (Run 2)

This project tests the performance of Quantum Machine Learning on High Energy Physics data. We use [CMS open data in miniaodsim format](https://opendata.cern.ch/search?page=1&size=20&experiment=CMS&year=2015&file_type=miniaodsim), and extract the data into root files with [POET-2015MiniAOD](https://github.com/cms-opendata-analyses/PhysObjectExtractorTool/tree/2015MiniAOD), following the guide from [CMS Open Data Workshop 2022](https://cms-opendata-workshop.github.io/2022-08-01-cms-open-data-workshop/).

### Prerequisites
1. A docker environment: See [CMSSW docker tutorial](https://cms-opendata-workshop.github.io/workshop2022-lesson-docker/)
2. Python3: Need packages such as [Awkward](https://awkward-array.org/quickstart.html), [uproot](https://uproot.readthedocs.io/en/latest/).

### Instructions
Two main workspaces:
1. LocalWorkspace: To be run in your local.
    - `jet.py`: Customly defined jet events class.
    - `demo_jet_info.ipynb`: Demonstration to usage of `jet.JetEvents`, and also see the information of data.
2. QCD_Jet_Fatjet: To be run in the docker.
    - `plugins/Analyzer.cc`: Analyzer for extracting hep data.
    - `python/ConfFile_cfg.py`: To be run with command `cmsRun`.
    - `root_files/download.py`: Can efficiently download with many channels.