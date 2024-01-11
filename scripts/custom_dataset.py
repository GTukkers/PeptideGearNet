'''
Original by: Danill Lepikhov
Functions to build the graphs
'''

# Import necessary libraries and modules
from torchdrug import data
import pickle
import tqdm
import torch as t
from torch import nn
import h5py
import torch.multiprocessing as mp
import time
from collections.abc import Mapping, Sequence
from joblib import Parallel, delayed
from rdkit import Chem
import os

# Define a Multilayer Perceptron (MLP) class using PyTorch
class MLP(nn.Module):
    def __init__(self, input_shape=3072, output=1,dropout_rate=.5):
        super(MLP,self).__init__()

        self.layers = nn.Sequential(
            # Input layer
            nn.Flatten(),
            nn.BatchNorm1d(input_shape*9),
            # Hidden layers
            nn.Linear(input_shape*9, input_shape),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(input_shape,input_shape),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(input_shape, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Output layer
            nn.Linear(1024, output),
        )

    def forward(self, x):
        return self.layers(x)

# Function to load Protein Data Bank (PDB) files and build a graph        
def load_pdb(pdb_file):
    """
    Build a graph from a pdb file.

    Parameters:
        pdb_files (list of str): pdb file names
    """
    mol = Chem.MolFromPDBFile(f"/{pdb_file}")
    pdb_name = pdb_file.split("/")[-3]
    if not mol:
        return # print(f"Unable to create molecule from PDB file {pdb_name}")
    protein = data.Protein.from_molecule(mol)
    if not protein:
        return # print(f"Unable to create protein from PDB file {pdb_name}")
    if hasattr(protein, "residue_feature"):
        with protein.residue():
            protein.residue_feature = protein.residue_feature.to_sparse()
    protein.pdb_path = pdb_file
    return protein, print(f"PDB file {pdb_name}")

# Define a custom ProteinDataset class based on torchdrug.data.ProteinDataset
class ProteinDataset(data.ProteinDataset):
    def __init__(self, build_h5 = False, h5_path = None, transform = None, 
        verbose = False, pdb_paths = None, subset = None,
        lazy=False, ntasks = 20, targets = None):
        # Initialize parameters and attributes
        self.ntasks = ntasks
        self.lazy = lazy
        self.transform = transform
        self.verbose = verbose
        self.h5 = h5_path
        missing_ids = []

        # Build graphs from PDB files if requested
        if build_h5:
            pdb_paths = tqdm.tqdm(pdb_paths, "Building graphs from pdb files")
            d = Parallel(n_jobs=ntasks)(delayed(load_pdb)(p) for p in pdb_paths) 
            print("loading done")
            self.data = [g for g in d if g is not None]
            self.failed_graph = [g for g in d if g is None]
            targets = [targets[i] for i in range(len(d)) if d[i] is not None]
            self.targets = {}
            self.targets["targets"] = targets
        else:
            # Load graphs from pre-built H5 file
            with h5py.File(h5_path) as h5_f:
                self.modelled_ids = [modeled_id.decode("utf8") for modeled_id in list(h5_f["ids"][:])] # ids of modelled cases
                if subset is not None:
                    ids = [i for i in subset if i in self.modelled_ids]
                    for i in subset:
                        if i not in self.modelled_ids:
                            # print(f"ID {i} is not present in the hdf5 file.")
                            missing_ids.append(i)
            if lazy == False:
                self.load_h5s(ids)
            else:
                self.h5_path = h5_path
                self.data = ids
            self.transform = transform

    # Function to load H5 graphs and targets
    def load_h5s(self, ids):
        if type(ids) == list:
            pool = mp.Pool(self.ntasks)
            graph_ids = ids
            target_ids = ids
            if self.verbose == True:
                graph_ids = tqdm.tqdm(ids, f"Loading case graphs")
                target_ids = tqdm.tqdm(ids, f"Loading case targets")
            proteins = data.Protein.pack(pool.map(self.load_h5_graph, graph_ids))
            targets = t.stack([self.load_h5_targets(case_id) for case_id in target_ids])
        else: # only one id is provided
            proteins = self.load_h5_graph(ids)
            targets = self.load_h5_targets(ids)
        if self.lazy == False:
            self.data = proteins
            self.targets = {"targets": targets}
        else:
            return proteins, targets
         
    # Function to load a single H5 graph     
    def load_h5_graph(self, case_id):
        with h5py.File(self.h5) as h5_f:
            h5 = h5_f[f"/graphs/{case_id}"]
            edge_list = t.as_tensor(h5["edge_list"][()])
            atom_type = t.as_tensor(h5["atom_type"][()])
            bond_type = t.as_tensor(h5["bond_type"][()])
            num_node = t.as_tensor(h5["num_node"][()])
            num_residue = t.as_tensor(h5["num_residue"][()])
            node_position = t.as_tensor(h5["node_position"][()])
            atom2residue = t.as_tensor(h5["atom2residue"][()])
            residue_feature = t.as_tensor(h5["residue_feature"][()])
            residue_type = t.as_tensor(h5["residue_type"][()])
            atom_name = t.as_tensor(h5["atom_name"][()])
            chain_id = t.as_tensor(h5["chain_id"][()])
        protein =  data.Protein(edge_list, atom_type, bond_type, num_node = num_node,
            num_residue = num_residue, node_position = node_position,
            atom_name = atom_name, residue_type = residue_type,
            residue_feature = residue_feature, chain_id = chain_id,
            atom2residue = atom2residue
        )
        # if len(case_id) == 1:
        #     with protein.residue():
        #         protein.residue_feature = protein.residue_feature.to_sparse()
        return protein

    # Function to load H5 targets
    def load_h5_targets(self, case_id):
        with h5py.File(self.h5) as h5_f:
            target = t.as_tensor(h5_f[f"targets/{case_id}"])
        return target
    
    # Function to save graphs and targets into H5 file
    def save_h5(self, h5_path):
        with h5py.File(h5_path, "w") as h5:
            indices = range(len(self.data))
            if self.verbose:
                indices = tqdm.tqdm(indices, f"Saving graphs into {self.h5}")
            for i in indices:
                graph = self.data[i]
                case_id = graph.pdb_path.split('/')[-1].replace('.pdb', '')
                h5_grp = h5.create_group(f"/graphs/{case_id}")
                h5_grp.create_dataset("edge_list", data = graph.edge_list)
                h5_grp.create_dataset("atom_type", data = graph.atom_type)
                h5_grp.create_dataset("bond_type", data = graph.bond_type)
                h5_grp.create_dataset("num_node", data = graph.num_node)
                h5_grp.create_dataset("num_residue", data = graph.num_residue)
                h5_grp.create_dataset("node_position", data = graph.node_position)
                h5_grp.create_dataset("atom2residue", data = graph.atom2residue)
                h5_grp.create_dataset("residue_feature", data = graph.residue_feature.to_dense())
                h5_grp.create_dataset("residue_type", data = graph.residue_type)
                h5_grp.create_dataset("atom_name", data = graph.atom_name)
                h5_grp.create_dataset("chain_id", data = graph.chain_id)

                targets = self.targets["targets"][i]
                h5.create_dataset(f"/targets/{case_id}", data = targets)
            h5.create_dataset(f"/ids", data=list(h5["targets"].keys()))
    
    # Function to get item from the dataset
    def get_item(self, index):
        if self.lazy:
            protein, targets = self.load_h5s(
                self.data[index] # the data is the list of case_ids when loaded lazy
            )
        else:
            protein = self.data[index].clone()
            targets = self.targets["targets"][index]
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        item["targets"] = targets
        item["id"] = self.data[index]
        return item