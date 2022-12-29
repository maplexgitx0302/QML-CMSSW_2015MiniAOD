import FWCore.ParameterSet.Config as cms

main_path = "/code/CMSSW_7_6_7/src/QCD_Jet_Fatjet/Analyzer/root_files/"

channel = "QCD_HT2000toInf"
num_events = 500

rootfile_path = [
    "file:"+main_path+"04A66FE4-47B9-E511-85C6-002590DB9216.root",
]

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(num_events) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(*rootfile_path)
)

from PhysicsTools.SelectorUtils.pfJetIDSelector_cfi import pfJetIDSelector

#----- Apply the noise jet ID filter -----#
process.looseAK4Jets = cms.EDFilter("PFJetIDSelectionFunctorFilter",
                                    filterParams = pfJetIDSelector.clone(),
                                    src = cms.InputTag("slimmedJets"))

process.looseAK8Jets = cms.EDFilter("PFJetIDSelectionFunctorFilter",
                                    filterParams = pfJetIDSelector.clone(),
                                    src = cms.InputTag("slimmedJetsAK8"))

#----- Configure the POET jet analyzers -----#
process.jets = cms.EDAnalyzer('Analyzer', 
				jets = cms.InputTag("slimmedJets"),
                fatjets = cms.InputTag("slimmedJetsAK8"),
				)

#----- RUN THE JOB! -----#
root_file_name = main_path + channel + "_" + str(num_events) + ".root"
process.TFileService = cms.Service("TFileService", fileName=cms.string(root_file_name))

process.p = cms.Path(process.jets)