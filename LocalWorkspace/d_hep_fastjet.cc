// fastjet
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"

// root
#include "TFile.h"
#include "TTree.h"
#include "TDirectory.h"

// standard library
#include <iostream>
#include <vector> 

int main(){
  std::string channel    = "ZprimeToZhToZlephbb";
  std::string num_events = "10000";
  std::string src_path   = "/home/yianchen/CMS_Open_Data_Workspace/CMSSW_7_6_7/src";
  std::string root_path  = src_path + "/QCD_Jet_Fatjet/Analyzer/root_files/";
  root_path += channel + "_" + num_events + ".root";

  std::unique_ptr<TFile> myFile( TFile::Open(root_path.c_str()) );
  TDirectory* dir = myFile->GetDirectory("jets");
  TTree* mtree = dir->Get<TTree>("Events");

  // std::vector<float> *fatjet_e  = 0;
  // std::vector<float> *fatjet_px = 0;
  // std::vector<float> *fatjet_py = 0;
  // std::vector<float> *fatjet_pz = 0;
  std::vector<std::vector<float>> *fatjet_daughter_e  = 0;
  TBranch *b_fatjet_daughter_e = 0;
  // std::vector<std::vector<float>> *fatjet_daughter_px = 0;
  // std::vector<std::vector<float>> *fatjet_daughter_py = 0;
  // std::vector<std::vector<float>> *fatjet_daughter_pz = 0;

  // mtree->ls();
  // mtree->SetBranchAddress("fatjet_e", &fatjet_e);
  // mtree->SetBranchAddress("fatjet_px",&fatjet_px);
  // mtree->SetBranchAddress("fatjet_py",&fatjet_py);
  // mtree->SetBranchAddress("fatjet_pz",&fatjet_pz);
  mtree->SetBranchAddress("fatjet_daughter_e", &fatjet_daughter_e, &b_fatjet_daughter_e);
  // mtree->SetBranchAddress("fatjet_daughter_px",&fatjet_daughter_px);
  // mtree->SetBranchAddress("fatjet_daughter_py",&fatjet_daughter_py);
  // mtree->SetBranchAddress("fatjet_daughter_pz",&fatjet_daughter_pz);

  // for (int iEntry = 0; iEntry < 10; ++iEntry) {
  //   // Load the data for the given tree entry
  //   mtree->GetEntry(iEntry);
  //   // std::cout << fatjet_daughter_e->size() << std::endl;
  // }
}