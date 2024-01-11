'''
Created by: Gijs Tukkers
Input:
    - path to the hdf5 file
    - dataframe containing the ID and binding affinity
Output
    - dataframe containing the ID, embedding, and target
'''

import resource
import pandas as pd
import torch as t
from torch import nn
from sklearn.model_selection import train_test_split
from torchdrug import datasets, transforms, models, data, layers, tasks, core
from torchdrug.layers import geometry
from custom_dataset import ProteinDataset, MLP
from finetune_args import param
import time
import warnings
import h5py
from mlp_designs import Conv2_1_hidden_layer, Conv_1_hidden_layer, MLP_1_hidden_layer, Reduce_1_hidden_layer, Gearnet_MLP, TransposeLayer, Hidden_0, Hidden_1, Hidden_2, Hidden_3, Conv_1, Conv_1_hidden_2, Conv_2_hidden_1, Conv_2_hidden_2, MLP_Hidden_1, MLP_Hidden_2, MLP_Conv_1_Hidden_1, MLP_Conv_1_Hidden_2
import sys
import numpy as np

# Function to adjust file limit to fix "RuntimeError"
def adjust_file_limit():
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def suppress_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="__floordiv__ is deprecated")

# Function to set and return the device for computation
def set_device():
    device = (t.device("cpu"), t.device("cuda:0"))[t.cuda.is_available()]
    print(f"Device used: {device}")
    return device

# Function to load and preprocess dataframe
def load_and_preprocess_dataframe(path):
    df = pd.read_csv(path)
    df = df.loc[(df.peptide.str.len() == 9) & (df.allele == "HLA-A*02:01")]
    # ids = df.ID.tolist()
    df["binder"] = df.measurement_value.apply(lambda x: int(x < 500))
    return df

# Function to load datasets using ProteinDataset class
def load_datasets(hdf5_path, subset_ids, protein_view_transform):
    return ProteinDataset(
        h5_path=hdf5_path,
        subset=subset_ids,
        build_h5=False,
        transform=protein_view_transform,
        verbose=True,
        lazy=True,
    )

# Function to construct graph using layers.GraphConstruction
def construct_graph():
    return layers.GraphConstruction(
        node_layers=[
            geometry.AlphaCarbonNode()  # loading only the alpha-carbon of each molecule
        ],
        edge_layers=[
            geometry.SpatialEdge(radius=10, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2)
        ],
        edge_feature="gearnet",
    )

def input_neurons(concatenate):
    return 3072 if concatenate else 512

# Function to create GearNet model
def create_gearnet_model(concatenate):
    return models.GearNet(
        input_dim=21,
        hidden_dims=[512, 512, 512, 512, 512, 512],
        batch_norm=True,
        concat_hidden=concatenate,
        short_cut=True,
        readout="sum",
        num_relation=7,
        edge_input_dim=59,
        num_angle_bin=8
    )

# Function to set up the task
def setup_task(gearnet_edge, graph_construction_model):
    return tasks.MultipleBinaryClassification(
        gearnet_edge,
        task=(0, 1),
        graph_construction_model=graph_construction_model,
        num_mlp_layer=5,  # Can be removed I think, the MLP is set later to the module
    )

# Function to load weights, map to device, and set model to eval mode
def prepare_task_model(task, ss_weights, device, own_mlp=None):
    task.model.load_state_dict(ss_weights, strict=False)
    task.model.to(device)
    task.model.eval()

def create_data_loaders(dataset, batch_size, ntasks):
    return data.dataloader.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=ntasks,
        drop_last=False,  # Turn to false to make sure that I have all the ID
    )

def create_embedding(task, dataloader, device, input_size, train_gearnet_weights=None, dataframe=None):
    metrics = {
        "ID": [],
        "embedding": [],
        "target": [],
    }

    # Train epoch
    for item in enumerate(dataloader):
        graph = task.graph_construction_model(item[1]["graph"]).to(device)
        complexes = item[1]['id']

        target = [dataframe[dataframe['ID'] == compleks]['binder'].values[0] for compleks in complexes]
        target = t.Tensor(target).to(device).view(-1, 1)
        peptide_mask = graph.chain_id == 16
        if train_gearnet_weights:
            gcn_output = task.model(graph, graph.node_feature.float())
        else:
            with t.no_grad():
                gcn_output = task.model(graph, graph.node_feature.float())

        try:
            peptide_emb = gcn_output["node_feature"][peptide_mask].reshape(1, 9, input_size)
        except Exception as e:
            print("Error:", e)
            continue

        peptide_emb_np = peptide_emb.cpu().numpy()
        target_np = target.cpu().numpy()

        metrics["ID"].append(complexes)
        metrics["embedding"].append(peptide_emb_np)
        metrics["target"].append(target_np)

    df_embedding = pd.DataFrame.from_dict(metrics)
    print(len(df_embedding))

    return df_embedding

def main():
    adjust_file_limit()
    suppress_warnings()
    device = set_device()

    # Need to keep the current train and validation set, separate so that I can compare it with my current results
    df = load_and_preprocess_dataframe(param.db1_path)
    df_test = load_and_preprocess_dataframe(param.db1_test)

    merged_df = pd.concat([df, df_test], ignore_index=True)
    ids = merged_df.ID.tolist()
    print(f"amount of ID's {len(ids)}")

    protein_view_transform = transforms.ProteinView("residue")

    dataset = load_datasets(param.h5_path, ids, protein_view_transform)

    graph_construction_model = construct_graph()
    gearnet_edge = create_gearnet_model(param.concat_gearnet)
    task = setup_task(gearnet_edge, graph_construction_model)
    ss_weights = t.load(param.mc_path, map_location=device)

    input_size = input_neurons(param.concat_gearnet)
    print(f"inputsize={input_size}")
    print(f"concat_gearnet={param.concat_gearnet}")
    prepare_task_model(task, ss_weights, device)

    dataloader = create_data_loaders(dataset, 1, param.num_workers)
    print(f"length dataloader {len(dataloader)}")

    df_embedding = create_embedding(task, dataloader, device, input_size=input_size, train_gearnet_weights=param.with_gearnet_weights, dataframe=merged_df)
    print(len(df_embedding))
    saved_model = f"GearNet-Edge_embedding"
    df_embedding.to_pickle(f"{param.output_dir}/{saved_model}_{str(input_size)}.pkl")

    # at the end, we save the model and metrics which are later loaded in the "explore.ipynb" file
    # t.save(metricss, f"{param.output_dir}/{saved_model}")
    print(f"Saved model in {param.output_dir}/{saved_model}")

if __name__ == "__main__":
    main()
