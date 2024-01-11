'''
Original by: Danill Lepikhov
First attempt to finetune the gearnet-edge embedding with a MLP
'''

## Load necessary libraries
from torch_scatter import scatter
from torchdrug import datasets, transforms, models, data, layers, tasks, core
from torchdrug.layers import geometry
import torch as t # IMPORTANT: everywhere the torch package is used with `t` alias
from torch import nn
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import time
import sys
import os
sys.path.append(os.path.abspath("."))
from custom_dataset import ProteinDataset, MLP
from finetune_args import param
from mlp_designs import WeightedConcatNetwork, WeightedConcat, ConvEmbedding

# Here the variable "a" contains all arguments passed to this file (the list is available in 
# finetune_args.py)

# Lines of code to to fix the "RuntimeError: received 0 items of ancdata" which happens sometimes on deepops cluster:
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
#####

# Check that CUDA is being used:
device = (t.device("cpu"),t.device("cuda:0"))[t.cuda.is_available()]
print(f"Device used: {device}")

# Path of the HDF5 where the features of graphs are saved (this saves time and aleviate the 
# need to generate features on the fly)
hdf5_path = param.h5_path

# Weights of the torchdrug model
ss_weights = t.load(param.mc_path, map_location=t.device(device))

# Database containing the caseID, allele, peptide and label column
df = pd.read_csv(param.db1_path)
# We work only on 9-mer peptides from the HLA-A*02:01 allele
df = df.loc[(df.peptide.str.len() == 9) & (df.allele == "HLA-A*02:01")]
ids = df.ID.tolist()

# df["binder"] is used as the label (already loaded in the hdf5 file)
df["binder"] = df.measurement_value.apply(lambda x: int(x < 500))

# Random split of case ids (using function from the scikit package)
train_ids, validation_ids = train_test_split(ids, test_size=.2, stratify=df.binder, random_state=1)
print(f"len of train_ids: {len(train_ids)}")
print(f"len of validation_ids: {len(validation_ids)}")

# Parameter to provided to load the graph on residue-level
protein_view_transform = transforms.ProteinView("residue")

print("Loading train dataset:")
train_dataset = ProteinDataset( # custom dataset class, no need to modify here
    h5_path = hdf5_path, 
    subset = train_ids,
    build_h5=False,
    transform = protein_view_transform,
    verbose = True,
    lazy = True,
)
print("Loading validation dataset:")
validation_dataset = ProteinDataset(
    h5_path = hdf5_path, 
    subset = validation_ids,
    build_h5=False,
    transform = protein_view_transform,
    verbose = True,
    lazy = True,
)

graph_construction_model = layers.GraphConstruction(
    node_layers=[
        geometry.AlphaCarbonNode() # loading only the alpha-carbon of each molecule
    ], 
    edge_layers=[
        geometry.SpatialEdge(radius=10, min_distance=5), # Parameters now the same as in paper
        geometry.KNNEdge(k=10, min_distance=5),
        geometry.SequentialEdge(max_distance=2)
    ],
    edge_feature="gearnet",
)

gearnet_edge = models.GearNet( # architecture of the torchdrug model
    input_dim = 21,
    hidden_dims = [512, 512, 512, 512, 512, 512],
    batch_norm = True,
    concat_hidden= True,                                             
    short_cut = True,
    readout = "sum",
    num_relation = 7,
    edge_input_dim = 59,
    num_angle_bin = 8
)
task = tasks.MultipleBinaryClassification( # mildly usefull lines
    gearnet_edge,
    task = (0, 1),
    graph_construction_model = graph_construction_model,
    num_mlp_layer = 5,                                                        
)

# Weights of the torchdrug model are loaded into the architecture
task.model.load_state_dict(ss_weights, strict=False) # torchdrug model

# Map torchdrug model to CUDA
task.model.to(device)

# IMPORTANT: torchdrug model in eval() mode
task.model.eval()

# Using the right neural network
INPUT_NEURONS = 3072
HIDDEN_NEURONS = 3072
HIDDEN_NEURONS2 = 32
OUTPUT_NEURONS = 1
DROPOUT_RATE =  0.5

class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class neural_network_no_hidden_layer(nn.Module):
    def __init__(self, input_shape=3072, output=1,dropout_rate=.5):
        super(neural_network_no_hidden_layer,self).__init__()
    
        self.layers = nn.Sequential(nn.Flatten(),
            nn.Linear(9 * input_shape, input_shape),
            nn.BatchNorm1d(input_shape),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_shape,output)
        )
        
    def forward(self, x):
        return self.layers(x)

neural_network_1_hidden_layer = nn.Sequential(nn.Flatten(),
                               nn.Linear(9 * INPUT_NEURONS, HIDDEN_NEURONS),
                               nn.BatchNorm1d(HIDDEN_NEURONS),
                               nn.ReLU(),
                               nn.Dropout(DROPOUT_RATE),
                               nn.Linear(HIDDEN_NEURONS, OUTPUT_NEURONS),
                               nn.ReLU(),
                               nn.Dropout(DROPOUT_RATE))

neural_network_2_hidden_layer = nn.Sequential(nn.Flatten(),
                               nn.Linear(9 * INPUT_NEURONS, 3* HIDDEN_NEURONS),
                               nn.BatchNorm1d(3 * HIDDEN_NEURONS),
                               nn.ReLU(),
                               nn.Dropout(DROPOUT_RATE),
                               nn.Linear(3 * HIDDEN_NEURONS, HIDDEN_NEURONS),
                               nn.ReLU(),
                               nn.Dropout(DROPOUT_RATE),
                               nn.Linear(HIDDEN_NEURONS, OUTPUT_NEURONS),
                               nn.ReLU(),
                               nn.Dropout(DROPOUT_RATE))

neural_network_1_conv = nn.Sequential(TransposeLayer(1,2),
                                      nn.Conv1d(INPUT_NEURONS, HIDDEN_NEURONS, 1),
                                      nn.BatchNorm1d(HIDDEN_NEURONS),
                                      nn.ReLU(),
                                      nn.Dropout(DROPOUT_RATE),
                                      nn.Flatten(),
                                      nn.Linear(9 * HIDDEN_NEURONS, OUTPUT_NEURONS),
                                      nn.ReLU(),
                                      nn.Dropout(DROPOUT_RATE))

neural_network_2_conv = nn.Sequential(TransposeLayer(1,2),
                                      nn.Conv1d(INPUT_NEURONS, HIDDEN_NEURONS, 1),
                                      nn.BatchNorm1d(HIDDEN_NEURONS),
                                      nn.ReLU(),
                                      nn.Dropout(DROPOUT_RATE),
                                      nn.Conv1d(HIDDEN_NEURONS, HIDDEN_NEURONS2, 1),
                                      nn.ReLU(),
                                      nn.Dropout(DROPOUT_RATE),
                                      nn.Flatten(),
                                      nn.Linear(9 * HIDDEN_NEURONS2, OUTPUT_NEURONS),
                                      nn.ReLU(),
                                      nn.Dropout(DROPOUT_RATE))

# loading the custom MLP from custom_dataset.py
# Maping to CUDA. The eval() and train() mode depends on train or inference
task.mlp = MLP()                                       
task.mlp.to(device)

# hyperparameters
batch_size = param.batch_size
ntasks = param.num_workers
epochs = param.epochs

# Params contains the weights of the model to optimize. It's either only MLP weights or MLP+torchdrug model weights. 
# By default, only the MLP weights are optimized.
params = (task.mlp.parameters(), task.parameters())[param.with_gearnet_weights]
optimizer = t.optim.Adam(params=params, lr = 1e-4, weight_decay=1e-5) # weight decay is applied here for L2 regularization
criterion = t.nn.BCELoss()
saved_model =f"Finetune_test_mlp_on_{param.output}_batch_{batch_size}_epochs_{epochs}" 
sigmoid = t.nn.Sigmoid()

all_losses = []
all_e_losses = []
all_targets = []
all_e_targets = []
all_pred = []
all_e_pred = []
all_times = []

best_model = {
    "model": None,
    "lowest_loss": 10000,
    "epoch": None,
}
final_model = {
    "model": None,
    "epoch": param.epochs,
    "batch_size": param.batch_size,
    "metrics": None
}

# train and evaluate on validation
train_dataloader = data.dataloader.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers = param.num_workers,
    drop_last = True,
)
validation_dataloader = data.dataloader.DataLoader(
    validation_dataset,
    batch_size = batch_size,
    num_workers = param.num_workers,
    drop_last = True,
)

# Those lines where used to verify that order of nodes is kept when graph is built:
# residue_vocab = ["G", "A", "S", "P", "V", "T", "C", "I", "L", "N",
#                  "D", "Q", "K", "E", "M", "H", "F", "R", "Y", "W"]
# batch = next(iter(train_dataloader))
# graph = task.graph_construction_model(batch["graph"])
# peptide_mask = graph.chain_id == 16
# print(f"Peptide length: {peptide_mask.nonzero().shape[0]}")
# print(f"Peptide: {df.loc[df.ID == batch['id'][0]].peptide}")
# peptide_seq = [residue_vocab[(res.long() == 1).nonzero()[0]] for res in graph.residue_feature[peptide_mask]]
# print(str(peptide_seq))

for e in range(epochs):
    print(f"starting training for epoch {e}")
    t1 = time.time()
    b = next(iter(train_dataloader))
    # train the model
    task.mlp.train()
    for (minib, batch) in enumerate(train_dataloader):
        graph = task.graph_construction_model(batch["graph"]).to(device) # generate the correct graph view from the batch
        target = batch["targets"].max(dim=1).indices.float().to(device).unsqueeze(1) # extract the batched labels
        peptide_mask = graph.chain_id == 16 # mask used to retrieve the peptide nodes only from the batched graph

        # if we plan to update the weights of the multiview contrastive model: 
        if param.with_gearnet_weights:
            gcn_output = task.model(graph, graph.node_feature.float())
        else:
            with t.no_grad():
                gcn_output = task.model(graph, graph.node_feature.float())
        try:
            peptide_emb = gcn_output["node_feature"][peptide_mask].reshape(batch_size, 9, INPUT_NEURONS)
        except:
            continue
        prop = sigmoid(task.mlp(peptide_emb))
        peptide_loss = criterion(prop, target)
        optimizer.zero_grad()
        peptide_loss.backward()
        optimizer.step()

        # if minib % 100 == 0:
        print(f"Training loss at minibatch {minib}: {peptide_loss:.5f}")

        # save stuff
        all_losses.append(peptide_loss)
        all_pred.append(prop)
        all_targets.append(target)

    # perform evaluation:
    task.mlp.eval()

    # Since the validation is batched (not all validation is run through the model at once),
    # we keep track of all losses in each batch to average them per epoch and have a loss value per epoch
    epoch_valid_loss = []
    for (minib, e_batch) in enumerate(validation_dataloader):
        e_graph = task.graph_construction_model(e_batch["graph"]).to(device)
        e_target = e_batch["targets"].max(dim=1).indices.float().to(device).unsqueeze(1)
        e_peptide_mask = e_graph.chain_id == 16
        with t.no_grad():
            e_gcn_output = task.model(e_graph, e_graph.node_feature.float())
            try:
                e_peptide_emb = e_gcn_output["node_feature"][e_peptide_mask].reshape(batch_size, 9, INPUT_NEURONS)
            except:
                continue
            e_prop = sigmoid(task.mlp(e_peptide_emb))
            e_peptide_loss = criterion(e_prop, e_target)
        if minib % 100 == 0:
            print(f"Validation loss at minibatch {minib}: {e_peptide_loss:.5f}")
        
        # add the loss to the epoch_valid_loss to calculate the mean loss later:
        epoch_valid_loss.append(e_peptide_loss)
                
        # Save values
        all_e_pred.append(e_prop)
        all_e_losses.append(e_peptide_loss)
        all_e_targets.append(e_target)
    epoch_valid_loss = t.stack(epoch_valid_loss)

    # If the average epoch loss is lower than the current best model loss, the model is saved as the best model
    if epoch_valid_loss.mean() < best_model["lowest_loss"]:
        best_model["model"] = task.mlp.state_dict()
        best_model["lowest_loss"] = epoch_valid_loss.mean()
        best_model["epoch"] = e
        t.save(best_model, f"{param.output_dir}/best_model_{saved_model}.pth")
        print(f"Saved best model at epoch {e} with loss {best_model['lowest_loss']}")

    t2 = time.time()
    print(f"Time for epoch (train and evaluation): {t2-t1} seconds")
    all_times.append(t2-t1)

to_pickle = {
"all_losses": all_losses,
"all_e_losses": all_e_losses,
"all_pred": all_pred,
"all_e_pred": all_e_pred,
"all_targets": all_targets,
"all_e_targets": all_e_targets,
"all_times": all_times
}
final_model["metrics"] = to_pickle
final_model["model"] = task.mlp.state_dict()

final_name = f"{param.output_dir}/{saved_model}.pth"

# at the end we save the model and metrics which are later loaded in the "explore.ipynb" file
t.save(final_model, final_name)
print(f"Saved model in {final_name}")