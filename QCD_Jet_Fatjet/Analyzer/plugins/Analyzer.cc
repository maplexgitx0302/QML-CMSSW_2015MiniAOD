// -*- C++ -*-
//
// Package:    ZprimeToZh_VS_QCD/Analyzer
// Class:      Analyzer
// 
/**\class Analyzer Analyzer.cc ZprimeToZh_VS_QCD/Analyzer/plugins/Analyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  
//         Created:  Fri, 07 Oct 2022 11:30:52 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "math.h"

//class to extract jet information
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

//classes to save data
#include "TTree.h"
#include "TFile.h"
#include<vector>

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class Analyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit Analyzer(const edm::ParameterSet&);
      ~Analyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<pat::JetCollection> jetToken_;
      edm::EDGetTokenT<pat::JetCollection> fatjetToken_;

      int numjet; //number of jets in the event
      int numfatjet; //number of jets in the event
      TTree *mtree;
      std::vector<float> jet_e;
      std::vector<float> jet_px;
      std::vector<float> jet_py;
      std::vector<float> jet_pz;
      std::vector<float> jet_pt;
      std::vector<float> jet_eta;
      std::vector<float> jet_phi;
      std::vector<float> jet_ch;
      std::vector<float> jet_mass;
      std::vector<std::vector<float>> jet_daughter_e;
      std::vector<std::vector<float>> jet_daughter_px;
      std::vector<std::vector<float>> jet_daughter_py;
      std::vector<std::vector<float>> jet_daughter_pz;
      std::vector<std::vector<float>> jet_daughter_pt;
      std::vector<std::vector<float>> jet_daughter_eta;
      std::vector<std::vector<float>> jet_daughter_phi;
      std::vector<std::vector<int>> jet_daughter_ch;
      std::vector<std::vector<float>> jet_daughter_mass;
      std::vector<std::vector<int>> jet_daughter_pdgid;

      std::vector<float> fatjet_e;
      std::vector<float> fatjet_px;
      std::vector<float> fatjet_py;
      std::vector<float> fatjet_pz;
      std::vector<float> fatjet_pt;
      std::vector<float> fatjet_eta;
      std::vector<float> fatjet_phi;
      std::vector<float> fatjet_ch;
      std::vector<float> fatjet_mass;
      std::vector<std::vector<float>> fatjet_daughter_e;
      std::vector<std::vector<float>> fatjet_daughter_px;
      std::vector<std::vector<float>> fatjet_daughter_py;
      std::vector<std::vector<float>> fatjet_daughter_pz;
      std::vector<std::vector<float>> fatjet_daughter_pt;
      std::vector<std::vector<float>> fatjet_daughter_eta;
      std::vector<std::vector<float>> fatjet_daughter_phi;
      std::vector<std::vector<int>> fatjet_daughter_ch;
      std::vector<std::vector<float>> fatjet_daughter_mass;
      std::vector<std::vector<int>> fatjet_daughter_pdgid;
      std::vector<float> fatjet_tau1;
      std::vector<float> fatjet_tau2;
      std::vector<float> fatjet_tau3;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Analyzer::Analyzer(const edm::ParameterSet& iConfig):
   jetToken_(consumes<pat::JetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
   fatjetToken_(consumes<pat::JetCollection>(iConfig.getParameter<edm::InputTag>("fatjets")))
{
   //now do what ever initialization is needed
   edm::Service<TFileService> fs;
   mtree = fs->make<TTree>("Events", "Events");

   mtree->Branch("numberjet",&numjet);
   mtree->GetBranch("numberjet")->SetTitle("Number of Jets");
   mtree->Branch("jet_e",&jet_e);
   mtree->GetBranch("jet_e")->SetTitle("Uncorrected Jet energy");
   mtree->Branch("jet_px",&jet_px);
   mtree->GetBranch("jet_px")->SetTitle("Uncorrected Jet Momentum x");
   mtree->Branch("jet_py",&jet_py);
   mtree->GetBranch("jet_py")->SetTitle("Uncorrected Jet Momentum y");
   mtree->Branch("jet_pz",&jet_pz);
   mtree->GetBranch("jet_pz")->SetTitle("Uncorrected Jet Momentum z");
   mtree->Branch("jet_pt",&jet_pt);
   mtree->GetBranch("jet_pt")->SetTitle("Uncorrected Transverse Jet Momentum");
   mtree->Branch("jet_eta",&jet_eta);
   mtree->GetBranch("jet_eta")->SetTitle("Jet Eta");
   mtree->Branch("jet_phi",&jet_phi);
   mtree->GetBranch("jet_phi")->SetTitle("Jet Phi");
   mtree->Branch("jet_ch",&jet_ch);
   mtree->GetBranch("jet_ch")->SetTitle("Jet Charge");
   mtree->Branch("jet_mass",&jet_mass);
   mtree->GetBranch("jet_mass")->SetTitle("Jet Mass");
   mtree->Branch("jet_px",&jet_px);
   mtree->GetBranch("jet_px")->SetTitle("Jet px");
   mtree->Branch("jet_daughter_e",&jet_daughter_e);
   mtree->GetBranch("jet_daughter_e")->SetTitle("Uncorrected Jet Daughter energy");
   mtree->Branch("jet_daughter_px",&jet_daughter_px);
   mtree->GetBranch("jet_daughter_px")->SetTitle("Uncorrected Jet Daughter Momentum x");
   mtree->Branch("jet_daughter_py",&jet_daughter_py);
   mtree->GetBranch("jet_daughter_py")->SetTitle("Uncorrected Jet Daughter Momentum y");
   mtree->Branch("jet_daughter_pz",&jet_daughter_pz);
   mtree->GetBranch("jet_daughter_pz")->SetTitle("Uncorrected Jet Daughter Momentum z");
   mtree->Branch("jet_daughter_pt",&jet_daughter_pt);
   mtree->GetBranch("jet_daughter_pt")->SetTitle("Uncorrected Transverse Jet Daughter Momentum");
   mtree->Branch("jet_daughter_eta",&jet_daughter_eta);
   mtree->GetBranch("jet_daughter_eta")->SetTitle("Jet Daughter Eta");
   mtree->Branch("jet_daughter_phi",&jet_daughter_phi);
   mtree->GetBranch("jet_daughter_phi")->SetTitle("Jet Daughter Phi");
   mtree->Branch("jet_daughter_ch",&jet_daughter_ch);
   mtree->GetBranch("jet_daughter_ch")->SetTitle("Jet Daughter Charge");
   mtree->Branch("jet_daughter_mass",&jet_daughter_mass);
   mtree->GetBranch("jet_daughter_mass")->SetTitle("Jet Daughter Mass");
   mtree->Branch("jet_daughter_pdgid",&jet_daughter_pdgid);
   mtree->GetBranch("jet_daughter_pdgid")->SetTitle("Jet Daughter PdgID");

   mtree->Branch("numberfatjet",&numfatjet);
   mtree->GetBranch("numberfatjet")->SetTitle("Number of Fatjets");
   mtree->Branch("fatjet_e",&fatjet_e);
   mtree->GetBranch("fatjet_e")->SetTitle("Uncorrected Fatjet energy");
   mtree->Branch("fatjet_px",&fatjet_px);
   mtree->GetBranch("fatjet_px")->SetTitle("Uncorrected Fatjet Momentum x");
   mtree->Branch("fatjet_py",&fatjet_py);
   mtree->GetBranch("fatjet_py")->SetTitle("Uncorrected Fatjet Momentum y");
   mtree->Branch("fatjet_pz",&fatjet_pz);
   mtree->GetBranch("fatjet_pz")->SetTitle("Uncorrected Fatjet Momentum z");
   mtree->Branch("fatjet_pt",&fatjet_pt);
   mtree->GetBranch("fatjet_pt")->SetTitle("Uncorrected Transverse Fatjet Momentum");
   mtree->Branch("fatjet_eta",&fatjet_eta);
   mtree->GetBranch("fatjet_eta")->SetTitle("Fatjet Eta");
   mtree->Branch("fatjet_phi",&fatjet_phi);
   mtree->GetBranch("fatjet_phi")->SetTitle("Fatjet Phi");
   mtree->Branch("fatjet_ch",&fatjet_ch);
   mtree->GetBranch("fatjet_ch")->SetTitle("Fatjet Charge");
   mtree->Branch("fatjet_mass",&fatjet_mass);
   mtree->GetBranch("fatjet_mass")->SetTitle("Fatjet Mass");
   mtree->Branch("fatjet_daughter_e",&fatjet_daughter_e);
   mtree->GetBranch("fatjet_daughter_e")->SetTitle("Uncorrected Fatjet Daughter energy");
   mtree->Branch("fatjet_daughter_px",&fatjet_daughter_px);
   mtree->GetBranch("fatjet_daughter_px")->SetTitle("Uncorrected Fatjet Daughter Momentum x");
   mtree->Branch("fatjet_daughter_py",&fatjet_daughter_py);
   mtree->GetBranch("fatjet_daughter_py")->SetTitle("Uncorrected Fatjet Daughter Momentum y");
   mtree->Branch("fatjet_daughter_pz",&fatjet_daughter_pz);
   mtree->GetBranch("fatjet_daughter_pz")->SetTitle("Uncorrected Fatjet Daughter Momentum z");
   mtree->Branch("fatjet_daughter_pt",&fatjet_daughter_pt);
   mtree->GetBranch("fatjet_daughter_pt")->SetTitle("Uncorrected Transverse Fatjet Daughter Momentum");
   mtree->Branch("fatjet_daughter_eta",&fatjet_daughter_eta);
   mtree->GetBranch("fatjet_daughter_eta")->SetTitle("Fatjet Daughter Eta");
   mtree->Branch("fatjet_daughter_phi",&fatjet_daughter_phi);
   mtree->GetBranch("fatjet_daughter_phi")->SetTitle("Fatjet Daughter Phi");
   mtree->Branch("fatjet_daughter_ch",&fatjet_daughter_ch);
   mtree->GetBranch("fatjet_daughter_ch")->SetTitle("Fatjet Daughter Charge");
   mtree->Branch("fatjet_daughter_mass",&fatjet_daughter_mass);
   mtree->GetBranch("fatjet_daughter_mass")->SetTitle("Fatjet Daughter Mass");
   mtree->Branch("fatjet_daughter_pdgid",&fatjet_daughter_pdgid);
   mtree->GetBranch("fatjet_daughter_pdgid")->SetTitle("Fatjet Daughter PdgID");
   mtree->Branch("fatjet_tau1",&fatjet_tau1);
   mtree->GetBranch("fatjet_tau1")->SetTitle("N-subjettiness tau_1 of Fatjet");
   mtree->Branch("fatjet_tau2",&fatjet_tau2);
   mtree->GetBranch("fatjet_tau2")->SetTitle("N-subjettiness tau_2 of Fatjet");
   mtree->Branch("fatjet_tau3",&fatjet_tau3);
   mtree->GetBranch("fatjet_tau3")->SetTitle("N-subjettiness tau_3 of Fatjet");

}


Analyzer::~Analyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
Analyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<pat::JetCollection> jets;
   iEvent.getByToken(jetToken_, jets);
   Handle<pat::JetCollection> fatjets;
   iEvent.getByToken(fatjetToken_, fatjets);

   numjet = 0;
   jet_e.clear();
   jet_px.clear();
   jet_py.clear();
   jet_pz.clear();
   jet_pt.clear();
   jet_eta.clear();
   jet_phi.clear();
   jet_ch.clear();
   jet_mass.clear();
   jet_px.clear();
   jet_daughter_e.clear();
   jet_daughter_px.clear();
   jet_daughter_py.clear();
   jet_daughter_pz.clear();
   jet_daughter_pt.clear();
   jet_daughter_eta.clear();
   jet_daughter_phi.clear();
   jet_daughter_ch.clear();
   jet_daughter_mass.clear();
   jet_daughter_pdgid.clear();

   numfatjet = 0;
   fatjet_e.clear();
   fatjet_px.clear();
   fatjet_py.clear();
   fatjet_pz.clear();
   fatjet_pt.clear();
   fatjet_eta.clear();
   fatjet_phi.clear();
   fatjet_ch.clear();
   fatjet_mass.clear();
   fatjet_daughter_e.clear();
   fatjet_daughter_px.clear();
   fatjet_daughter_py.clear();
   fatjet_daughter_pz.clear();
   fatjet_daughter_pt.clear();
   fatjet_daughter_eta.clear();
   fatjet_daughter_phi.clear();
   fatjet_daughter_ch.clear();
   fatjet_daughter_mass.clear();
   fatjet_daughter_pdgid.clear();
   fatjet_tau1.clear();
   fatjet_tau2.clear();
   fatjet_tau3.clear();

   if(jets.isValid()){
      for (const pat::Jet &jet : *jets){
         pat::Jet uncorrJet = jet.correctedJet(0);
         jet_e.push_back(uncorrJet.energy());
         jet_px.push_back(uncorrJet.px());
         jet_py.push_back(uncorrJet.py());
         jet_pz.push_back(uncorrJet.pz());
         jet_pt.push_back(uncorrJet.pt());
         jet_eta.push_back(uncorrJet.eta());
         jet_phi.push_back(uncorrJet.phi());
         jet_ch.push_back(uncorrJet.charge());
         jet_mass.push_back(uncorrJet.mass());
         jet_px.push_back(uncorrJet.px());
         jet_daughter_e.push_back({});
         jet_daughter_px.push_back({});
         jet_daughter_py.push_back({});
         jet_daughter_pz.push_back({});
         jet_daughter_pt.push_back({});
         jet_daughter_eta.push_back({});
         jet_daughter_phi.push_back({});
         jet_daughter_ch.push_back({});
         jet_daughter_mass.push_back({});
         jet_daughter_pdgid.push_back({});
         ++numjet;
         const size_t num_daughter = jet.numberOfDaughters();
         for(size_t d=0; d<num_daughter; d++){
            const reco::Candidate * Daughter = jet.daughter(d);
            jet_daughter_e.back().push_back(Daughter->energy());
            jet_daughter_px.back().push_back(Daughter->px());
            jet_daughter_py.back().push_back(Daughter->py());
            jet_daughter_pz.back().push_back(Daughter->pz());
            jet_daughter_pt.back().push_back(Daughter->pt());
            jet_daughter_eta.back().push_back(Daughter->eta());
            jet_daughter_phi.back().push_back(Daughter->phi());
            jet_daughter_ch.back().push_back(Daughter->charge());
            jet_daughter_mass.back().push_back(Daughter->mass());
            jet_daughter_pdgid.back().push_back(Daughter->pdgId());
         }
      } 
   }

   if(fatjets.isValid()){
      for (const pat::Jet &fatjet : *fatjets){
         pat::Jet uncorrFatjet = fatjet.correctedJet(0);
         fatjet_e.push_back(uncorrFatjet.energy());
         fatjet_px.push_back(uncorrFatjet.px());
         fatjet_py.push_back(uncorrFatjet.py());
         fatjet_pz.push_back(uncorrFatjet.pz());
         fatjet_pt.push_back(uncorrFatjet.pt());
         fatjet_eta.push_back(uncorrFatjet.eta());
         fatjet_phi.push_back(uncorrFatjet.phi());
         fatjet_ch.push_back(uncorrFatjet.charge());
         fatjet_mass.push_back(uncorrFatjet.mass());
         fatjet_daughter_e.push_back({});
         fatjet_daughter_px.push_back({});
         fatjet_daughter_py.push_back({});
         fatjet_daughter_pz.push_back({});
         fatjet_daughter_pt.push_back({});
         fatjet_daughter_eta.push_back({});
         fatjet_daughter_phi.push_back({});
         fatjet_daughter_ch.push_back({});
         fatjet_daughter_mass.push_back({});
         fatjet_daughter_pdgid.push_back({});
         fatjet_tau1.push_back((double)uncorrFatjet.userFloat("NjettinessAK8:tau1"));
         fatjet_tau2.push_back((double)uncorrFatjet.userFloat("NjettinessAK8:tau2"));
         fatjet_tau3.push_back((double)uncorrFatjet.userFloat("NjettinessAK8:tau3"));
         ++numfatjet;
         const size_t num_daughter = fatjet.numberOfDaughters();
         for(size_t d=0; d<num_daughter; d++){
            const reco::Candidate * Daughter = fatjet.daughter(d);
            if(Daughter->pdgId()==0){
               for(size_t gd=0; gd<Daughter->numberOfDaughters(); gd++){
                  const reco::Candidate * GrandDaughter = Daughter->daughter(gd);
                  fatjet_daughter_e.back().push_back(GrandDaughter->energy());
                  fatjet_daughter_px.back().push_back(GrandDaughter->px());
                  fatjet_daughter_py.back().push_back(GrandDaughter->py());
                  fatjet_daughter_pz.back().push_back(GrandDaughter->pz());
                  fatjet_daughter_pt.back().push_back(GrandDaughter->pt());
                  fatjet_daughter_eta.back().push_back(GrandDaughter->eta());
                  fatjet_daughter_phi.back().push_back(GrandDaughter->phi());
                  fatjet_daughter_ch.back().push_back(GrandDaughter->charge());
                  fatjet_daughter_mass.back().push_back(GrandDaughter->mass());
                  fatjet_daughter_pdgid.back().push_back(GrandDaughter->pdgId());
               }
            }else{
               fatjet_daughter_e.back().push_back(Daughter->energy());
               fatjet_daughter_px.back().push_back(Daughter->px());
               fatjet_daughter_py.back().push_back(Daughter->py());
               fatjet_daughter_pz.back().push_back(Daughter->pz());
               fatjet_daughter_pt.back().push_back(Daughter->pt());
               fatjet_daughter_eta.back().push_back(Daughter->eta());
               fatjet_daughter_phi.back().push_back(Daughter->phi());
               fatjet_daughter_ch.back().push_back(Daughter->charge());
               fatjet_daughter_mass.back().push_back(Daughter->mass());
               fatjet_daughter_pdgid.back().push_back(Daughter->pdgId());
            }
         }
      } 
   }

   mtree->Fill();
   return;
}


// ------------ method called once each job just before starting event loop  ------------
void 
Analyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
Analyzer::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
Analyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Analyzer);
