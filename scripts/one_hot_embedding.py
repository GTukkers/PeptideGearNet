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
# Load the MLPs that you've created
from mlp_designs import Reduce_to_20_concat_one_hot, Reduce_to_20, Gearnet_MLP, TransposeLayer, Hidden_0, Hidden_1, Hidden_2, Hidden_3, Conv_1, Conv_1_hidden_2, Conv_2_hidden_1, Conv_2_hidden_2, MLP_Hidden_1, MLP_Hidden_2, MLP_Conv_1_Hidden_1, MLP_Conv_1_Hidden_2, MLP_reduce_input, Reduce_and_onhot, Reduce_and_onhot_seq
import sys

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
    ids = df.ID.tolist()
    df["binder"] = df.measurement_value.apply(lambda x: int(x < 500))
    return df, ids

# Function to split data into training and validation sets
def split_data(df, ids):
    training_ids, validation_ids = train_test_split(ids, test_size=0.2, stratify=df.binder, random_state=1)
    print(f"len of train_ids: {len(training_ids)}\nlen of validation_ids: {len(validation_ids)}")
    return training_ids, validation_ids

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
        num_mlp_layer=5, 
    )

# Function to load weights, map to device, and set model to eval mode
def prepare_task_model(task, ss_weights, device, own_mlp=None):
    task.model.load_state_dict(ss_weights, strict=False)
    task.model.to(device)
    task.model.eval()
    task.mlp = own_mlp                                          
    task.mlp.to(device)

def create_data_loaders(dataset, batch_size, ntasks):
    return data.dataloader.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=ntasks,
        drop_last=True,
    )
def protein_to_onehot(seq):
    # Make an Identity matrix with 20 rows as to encode the 20 AAs 
    AAs_hot = t.eye(20, dtype=t.float)

    aminoacids = ('ACDEFGHIKLMNPQRSTVWY')
    return t.stack([AAs_hot[aminoacids.index(aa)]for aa in seq])

# Define your training function
def train_epoch(task, train_dataloader, device, optimizer, loss_function, train_metrics, input_size, train_gearnet_weights=None, batch_size=None, dataframe=None, second_nn=None):
    sequences_train = []
    for (minib, batch) in enumerate(train_dataloader):
        graph = task.graph_construction_model(batch["graph"]).to(device)
        peptide_mask = graph.chain_id == 16
        complexes = batch['id']
        target = [dataframe[dataframe['ID'] == compleks]['binder'].values[0] for compleks in complexes]
        target = t.Tensor(target).to(device).view(-1, 1)
        
        sequence = [dataframe[dataframe['ID'] == compleks]['peptide'].values[0] for compleks in complexes]

        if train_gearnet_weights:
            gcn_output = task.model(graph, graph.node_feature.float())
        else:
            with t.no_grad():
                gcn_output = task.model(graph, graph.node_feature.float())
        try:
            peptide_emb = gcn_output["node_feature"][peptide_mask].reshape(batch_size, 9, input_size)
        except Exception as e:
            print("Error:", e)
            continue
        
        predict = task.mlp(peptide_emb)
        one_hot =  t.stack([protein_to_onehot(seq) for seq in sequence]).to(device)
        combine = t.cat((predict, one_hot), dim=2).to(device)
        combine.to(device)
        predict = second_nn(combine)
        peptide_loss = loss_function(predict, target)
        
        
        optimizer.zero_grad()
        peptide_loss.backward()
        optimizer.step()

        if minib % 1 == 0:
            print(f"Training loss at minibatch {minib}: {peptide_loss:.5f}")
        
        train_metrics["losses"].append(peptide_loss)
        train_metrics["predict"].append(predict)
        train_metrics["targets"].append(target)


# Define your validation function
def validate_epoch(task, validation_dataloader, device, loss_function, val_metrics, input_size, batch_size=None, dataframe=None, second_nn=None):
    epoch_val_loss = []
    sequences_validation = []
    for (minib, batch) in enumerate(validation_dataloader):
        graph = task.graph_construction_model(batch["graph"]).to(device)
        peptide_mask = graph.chain_id == 16
        complexes = batch['id']
        target = [dataframe[dataframe['ID'] == compleks]['binder'].values[0] for compleks in complexes]
        sequence = [dataframe[dataframe['ID'] == compleks]['peptide'].values[0] for compleks in complexes]
        target = t.Tensor(target).to(device).view(-1, 1)

        
        with t.no_grad():
            gcn_output = task.model(graph, graph.node_feature.float())
            
        try:
            peptide_emb = gcn_output["node_feature"][peptide_mask].reshape(batch_size, 9, input_size)
        except Exception as e:
            print("Error:", e)
            continue
        
        predict = task.mlp(peptide_emb)
        one_hot =  t.stack([protein_to_onehot(seq) for seq in sequence]).to(device)

        combine = t.cat((predict, one_hot), dim=2).to(device) 
        combine.to(device)
        predict = second_nn(combine)
        peptide_loss = loss_function(predict, target)
        

        if minib % 1 == 0:
            print(f"Validation loss at minibatch {minib}: {peptide_loss:.5f}")
        
        epoch_val_loss.append(peptide_loss)
        val_metrics["losses"].append(peptide_loss)
        val_metrics["predict"].append(predict)
        val_metrics["targets"].append(target)
    epoch_val_loss = t.stack(epoch_val_loss)
    return epoch_val_loss

def train_model(task, train_dataloader, validation_dataloader, test_dataloader, device, optimizer, loss_function, epochs, input_size, train_gearnet_weights=None, batch_size=None, dataframe=None, second_nn=None):
    best_model = {
        "model": None,
        "lowest_loss": 10000,
        "epoch": None
    }
    final_model = {
        "model": None,
        "epoch": epochs,
        "batch_size": batch_size,
        "metrics": None
    }
    metrics = {
        "train": {"losses": [], "predict": [], "targets": []},
        "val": {"losses": [], "predict": [], "targets": []},
        "test": {"losses": [], "predict": [], "targets": []},
        "all_times": []
    }
    for epoch in range(epochs):
        print(f"starting training for epoch {epoch}")
        t1 = time.time()
        task.model.eval()
        task.mlp.eval()
        second_nn.train()
        train_epoch(task, train_dataloader, device, optimizer, loss_function, metrics["train"], input_size, train_gearnet_weights, batch_size, dataframe, second_nn)
        second_nn.eval()
        epoch_valid_loss = validate_epoch(task, validation_dataloader, device, loss_function, metrics["val"], input_size, batch_size, dataframe, second_nn)

        if epoch_valid_loss.mean() < best_model["lowest_loss"]:
            best_model["model"] = task.mlp.state_dict()
            best_model["lowest_loss"] = epoch_valid_loss.mean()
            best_model["epoch"] = epoch

        t2 = time.time()
        print(f"Time for epoch (train and evaluation): {t2-t1} seconds")
        metrics["all_times"].append(t2-t1)

    final_model["metrics"] = metrics
    final_model["model"] = task.mlp.state_dict()

    task.mlp.load_state_dict(best_model["model"])
    test_loss_best = validate_epoch(task, test_dataloader, device, loss_function, metrics["test"], input_size, batch_size, dataframe)
    best_model["test_loss"] = test_loss_best.mean()
    return final_model, best_model

def train_validate_model2(task, train_dataloader, validation_dataloader, test_dataloader, device, optimizer, loss_function, epochs, input_size, train_gearnet_weights=None, batch_size=None, dataframe=None, second_nn=None, dataframe_test=None):
    best_model = {
        "model": None,
        "lowest_loss": 10000,
        "epoch": None,
        "test_loss": None
    }
    final_model = {
        "model": None,
        "epoch": epochs,
        "batch_size": batch_size,
        "metrics": None
    }
    metrics = {
        "train": {"losses": [], "predict": [], "targets": []},
        "val": {"losses": [], "predict": [], "targets": []},
        "test": {"losses": [], "predict": [], "targets": []},
        "all_times": []
    }
    for epoch in range(epochs):
        print(f"starting training for epoch {epoch}")
        t1 = time.time()

        # Train epoch
        task.mlp.train()
        for (minib, batch) in enumerate(train_dataloader):
            graph = task.graph_construction_model(batch["graph"]).to(device)
            complexes = batch['id']
            target = [dataframe[dataframe['ID'] == compleks]['binder'].values[0] for compleks in complexes]
            target = t.Tensor(target).to(device).view(-1, 1)
            sequence = [dataframe[dataframe['ID'] == compleks]['peptide'].values[0] for compleks in complexes]
            peptide_mask = graph.chain_id == 16
            if train_gearnet_weights:
                gcn_output = task.model(graph, graph.node_feature.float())
            else:
                with t.no_grad():
                    gcn_output = task.model(graph, graph.node_feature.float())

            try:
                peptide_emb = gcn_output["node_feature"][peptide_mask].reshape(batch_size, 9, input_size)
            except Exception as e:
                print("Error:", e)
                continue
            
            predict = task.mlp(peptide_emb)
            one_hot =  t.stack([protein_to_onehot(seq) for seq in sequence]).to(device)
            combine = t.cat((predict, one_hot), dim=2).to(device)
            combine.to(device)
            predict = second_nn(combine)
            peptide_loss = loss_function(predict, target)

            optimizer.zero_grad()
            peptide_loss.backward()
            optimizer.step()

            if minib % 10 == 0:
                print(f"Training loss at minibatch {minib}: {peptide_loss:.5f}")
            
            metrics["train"]["losses"].append(peptide_loss)
            metrics["train"]["predict"].append(predict)
            metrics["train"]["targets"].append(target)
            del gcn_output
            del peptide_emb
            del predict

        # Validate epoch
        task.mlp.eval()
        epoch_val_loss = []

        for (minib, batch) in enumerate(validation_dataloader):
            graph = task.graph_construction_model(batch["graph"]).to(device)
            peptide_mask = graph.chain_id == 16
            complexes = batch['id']
            target = [dataframe[dataframe['ID'] == compleks]['binder'].values[0] for compleks in complexes]
            target = t.Tensor(target).to(device).view(-1, 1)
            sequence = [dataframe[dataframe['ID'] == compleks]['peptide'].values[0] for compleks in complexes]

            with t.no_grad():
                gcn_output = task.model(graph, graph.node_feature.float())
                
            try:
                peptide_emb = gcn_output["node_feature"][peptide_mask].reshape(batch_size, 9, input_size)
            except Exception as e:
                print("Error:", e)
                continue
            
            predict = task.mlp(peptide_emb)
            one_hot =  t.stack([protein_to_onehot(seq) for seq in sequence]).to(device)
            combine = t.cat((predict, one_hot), dim=2).to(device)
            combine.to(device)
            predict = second_nn(combine)
            peptide_loss = loss_function(predict, target)

            if minib % 10 == 0:
                print(f"Validation loss at minibatch {minib}: {peptide_loss:.5f}")
            
            epoch_val_loss.append(peptide_loss)
            metrics["val"]["losses"].append(peptide_loss)
            metrics["val"]["predict"].append(predict)
            metrics["val"]["targets"].append(target)
            del gcn_output
            del peptide_emb
            del predict

        epoch_val_loss = t.stack(epoch_val_loss)

        if epoch_val_loss.mean() < best_model["lowest_loss"]:
            best_model["model"] = task.mlp.state_dict()
            best_model["lowest_loss"] = epoch_val_loss.mean()
            best_model["epoch"] = epoch

        t2 = time.time()
        print(f"Time for epoch (train and evaluation): {t2-t1} seconds")
        metrics["all_times"].append(t2-t1)

    final_model["metrics"] = metrics
    final_model["model"] = task.mlp.state_dict()

    task.mlp.load_state_dict(best_model["model"])
    test_loss_best = []

    # Test epoch
    for (minib, batch) in enumerate(test_dataloader):
        graph = task.graph_construction_model(batch["graph"]).to(device)
        complexes = batch['id']
        target = [dataframe_test[dataframe_test['ID'] == compleks]['binder'].values[0] for compleks in complexes]
        target = t.Tensor(target).to(device).view(-1, 1)
        sequence = [dataframe_test[dataframe_test['ID'] == compleks]['peptide'].values[0] for compleks in complexes]
        peptide_mask = graph.chain_id == 16

        with t.no_grad():
            gcn_output = task.model(graph, graph.node_feature.float())
            
        try:
            peptide_emb = gcn_output["node_feature"][peptide_mask].reshape(batch_size, 9, input_size)
        except Exception as e:
            print("Error:", e)
            continue
        
        predict = task.mlp(peptide_emb)
        one_hot =  t.stack([protein_to_onehot(seq) for seq in sequence]).to(device)
        combine = t.cat((predict, one_hot), dim=2).to(device)
        combine.to(device)
        predict = second_nn(combine)
        peptide_loss = loss_function(predict, target)
        
        test_loss_best.append(peptide_loss)
        metrics["test"]["losses"].append(peptide_loss)
        metrics["test"]["predict"].append(predict)
        metrics["test"]["targets"].append(target)
        del gcn_output
        del peptide_emb
        del predict

    best_model["test_loss"] = t.stack(test_loss_best).mean()
    return final_model, best_model


def main():
    adjust_file_limit()
    suppress_warnings()
    device = set_device()

    df, ids = load_and_preprocess_dataframe(param.db1_path)
    df_test, test_ids = load_and_preprocess_dataframe(param.db1_test)
    print(f"len of test_ids: {len(test_ids)}")

    train_ids, validation_ids = split_data(df, ids)
    protein_view_transform = transforms.ProteinView("residue")
    
    train_dataset = load_datasets(param.h5_path, train_ids, protein_view_transform)
    validation_dataset = load_datasets(param.h5_path, validation_ids, protein_view_transform)
    test_dataset = load_datasets(param.h5_path, test_ids, protein_view_transform)

    graph_construction_model = construct_graph()
    gearnet_edge = create_gearnet_model(param.concat_gearnet)
    task = setup_task(gearnet_edge, graph_construction_model)
    ss_weights = t.load(param.mc_path, map_location=device) 

    input_size = input_neurons(param.concat_gearnet)
    print(f"inputsize={input_size}")
    print(f"concat_gearnet={param.concat_gearnet}")
    #Add own neural network
    neural_network = Reduce_to_20(input_shape=input_size, hidden1=param.hidden1, hidden2=param.hidden2, hidden3=param.hidden3, dropout_rate=param.dropout)
    prepare_task_model(task, ss_weights, device, neural_network)
    print(task.mlp)
    
    train_dataloader = create_data_loaders(train_dataset, param.batch_size, param.num_workers)
    validation_dataloader = create_data_loaders(validation_dataset, param.batch_size, param.num_workers)
    test_dataloader = create_data_loaders(test_dataset, param.batch_size, param.num_workers)

    loss_function = t.nn.BCELoss()
    #params = (task.mlp.parameters(), task.parameters())[param.with_gearnet_weights]
    second_mlp = Reduce_to_20_concat_one_hot(input_shape=input_size, hidden1=param.hidden1, hidden2=param.hidden2, hidden3=param.hidden3, dropout_rate=param.dropout)
    second_mlp.to(device)
    print(second_mlp)
    params = second_mlp.parameters() #tuple(params) + 
    optimizer = t.optim.Adam(params=params)#, lr=1e-4, weight_decay=1e-5)
    final_model, best_model = train_validate_model2(task, train_dataloader, validation_dataloader, test_dataloader, device, optimizer, loss_function, param.epochs, input_size=input_size, train_gearnet_weights=param.with_gearnet_weights, batch_size=param.batch_size, dataframe=df, second_nn=second_mlp, dataframe_test=df_test)
    saved_model = f"MLP_using_{param.output}_batch_{param.batch_size}_epochs_{param.epochs}"

    # at the end we save the model and metrics which are later loaded in the "explore.ipynb" file
    t.save(final_model, f"{param.output_dir}/{saved_model}.pth")
    print(f"Saved model in {param.output_dir}/{saved_model}.pth")
    t.save(best_model, f"{param.output_dir}/best_model_{saved_model}.pth")
    print(f"Saved best model at epoch {best_model['epoch']} with loss {best_model['lowest_loss']}")

if __name__ == "__main__":
    main()