import FWCore.ParameterSet.Config as cms

"""Configuration Parameters"""
MAX_EVENTS = 5000
# MAIN_PROCESS = "ZprimeToZhToZinvhbb"
# MAIN_PROCESS = "ZprimeToZhToZlephbb"
# MAIN_PROCESS = "QCD_HT1500to2000"
MAIN_PROCESS = "QCD_HT2000toInf"
"""------------------------"""

f = open("/code/CMSSW_7_6_7/src/ZprimeToZh_VS_QCD/Analyzer/python/Rootfiles_download/"+MAIN_PROCESS+".txt", "r")
rootfile_url = f.read().splitlines()
f.close()

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(MAX_EVENTS) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(*rootfile_url)
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
process.TFileService = cms.Service("TFileService", fileName=cms.string(MAIN_PROCESS+".root"))

process.p = cms.Path(process.jets)
