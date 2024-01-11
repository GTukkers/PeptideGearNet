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
from mlp_designs import Conv_1_hidden_layer, MLP_1_hidden_layer, Reduce_1_hidden_layer
import sys
from torch.utils.data import Dataset, DataLoader

class Embedding_Dataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return {
            'embedding': self.embeddings[index],
            'target': self.targets[index],
        }

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
# I think target can be added here
def load_datasets(hdf5_path, subset_ids, protein_view_transform):
    return ProteinDataset(
        h5_path=hdf5_path,
        subset=subset_ids,
        build_h5=False,
        transform=protein_view_transform,
        verbose=True,
        lazy=True,
    )

def load_embedding(path):
    return pd.read_pickle(path)

def split_embedding(data, train_id, val_id, test_id):
    data['ID'] = data['ID'].apply(lambda x: x[0])
    data['embedding'] = data['embedding'].apply(lambda x: x[0])
    data['target'] = data['target'].apply(lambda x: x[0])   
    train_data = data[data['ID'].isin(train_id)] 
    val_data = data[data['ID'].isin(val_id)] 
    test_data = data[data['ID'].isin(test_id)] 
    return train_data, val_data, test_data

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
        task=(0,1),
        graph_construction_model=graph_construction_model,
        num_mlp_layer=5, #Can be removed I think, the mlp is set later to the module
    )

# Function to load weights, map to device, and set model to eval mode
def prepare_task_model(task, ss_weights, device, own_mlp=None):
    task.model.load_state_dict(ss_weights, strict=False)
    task.model.to(device)
    task.model.eval()
    task.mlp = own_mlp                                          
    task.mlp.to(device)

def create_data_loaders(dataset, batch_size, num_workers):
    dataset_embedding = t.tensor(dataset.embedding.tolist())
    dataset_target = t.tensor(dataset.target.tolist())
    custom_dataset = Embedding_Dataset(dataset_embedding,dataset_target)
    dataloader = DataLoader(custom_dataset, batch_size= batch_size, num_workers=num_workers, drop_last=True)
    return dataloader

def train_validate_model2(neural_network, train_dataloader, validation_dataloader, test_dataloader, device, optimizer, loss_function, epochs, batch_size=None):
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
        neural_network.train()
        for (minib, batch) in enumerate(train_dataloader):
            embedding = batch['embedding'].to(device)
            target = batch['target'].to(device)
            predict = neural_network(embedding)
            peptide_loss = loss_function(predict, target)
            optimizer.zero_grad()
            peptide_loss.backward()
            optimizer.step()

            if minib % 100 == 0:
                print(f"Training loss at minibatch {minib}: {peptide_loss:.5f}")
            
            metrics["train"]["losses"].append(peptide_loss)
            metrics["train"]["predict"].append(predict)
            metrics["train"]["targets"].append(target)


        # Validate epoch
        neural_network.eval()
        epoch_val_loss = []

        for (minib, batch) in enumerate(validation_dataloader):
            embedding = batch['embedding'].to(device)
            target = batch['target'].to(device)
            predict = neural_network(embedding)
            peptide_loss = loss_function(predict, target)

            if minib % 100 == 0:
                print(f"Validation loss at minibatch {minib}: {peptide_loss:.5f}")
            
            epoch_val_loss.append(peptide_loss)
            metrics["val"]["losses"].append(peptide_loss)
            metrics["val"]["predict"].append(predict)
            metrics["val"]["targets"].append(target)


        epoch_val_loss = t.stack(epoch_val_loss)

        if epoch_val_loss.mean() < best_model["lowest_loss"]:
            best_model["model"] = neural_network.state_dict()
            best_model["lowest_loss"] = epoch_val_loss.mean()
            best_model["epoch"] = epoch

        t2 = time.time()
        print(f"Time for epoch (train and evaluation): {t2-t1} seconds")
        metrics["all_times"].append(t2-t1)

    final_model["metrics"] = metrics
    final_model["model"] = neural_network.state_dict()

    neural_network.load_state_dict(best_model["model"])
    test_loss_best = []

    # Test epoch
    for (minib, batch) in enumerate(test_dataloader):
        embedding = batch['embedding'].to(device) 
        target = batch['target'].to(device)
        predict = neural_network(embedding)
        peptide_loss = loss_function(predict, target)
        test_loss_best.append(peptide_loss)

        metrics["test"]["losses"].append(peptide_loss)
        metrics["test"]["predict"].append(predict)
        metrics["test"]["targets"].append(target)


    best_model["test_loss"] = t.stack(test_loss_best).mean()
    return final_model, best_model

def main():
    adjust_file_limit()
    suppress_warnings()
    device = set_device()

    # For shuffled dataset I need to know which ID's to choose
    # df, ids = load_and_preprocess_dataframe(param.db1_path)
    # df_test, test_ids = load_and_preprocess_dataframe(param.db1_test)
    # print(f"len of test_ids: {len(test_ids)}")
    # train_ids, validation_ids = split_data(df, ids)
    
    input_size = input_neurons(param.concat_gearnet)
    if param.concat_gearnet:
        embedding = load_embedding(param.embedding_3072)
    else:
        embedding = load_embedding(param.embedding_512)
    print(len(embedding))

    df = pd.read_csv(param.OOD)
    train_ids = df.loc[df['datatype'] == "train", 'ID'].tolist()
    validation_ids = df.loc[df['datatype'] == "val", 'ID'].tolist()
    test_ids = df.loc[df['datatype'] == "test", 'ID'].tolist()
    
    train_emb, val_emb, test_emb = split_embedding(embedding, train_ids, validation_ids, test_ids)

    train_dataloader = create_data_loaders(train_emb, param.batch_size, param.num_workers)
    validation_dataloader = create_data_loaders(val_emb, param.batch_size, param.num_workers)
    test_dataloader = create_data_loaders(test_emb, param.batch_size, param.num_workers)

    print(f"inputsize={input_size}")
    #Add own neural network
    neural_network = Reduce_1_hidden_layer(input_shape=input_size, hidden1=param.hidden1, hidden2=param.hidden2, hidden3=param.hidden3, dropout_rate=param.dropout)
    neural_network = neural_network.to(device)

    print(neural_network)
       
    loss_function = t.nn.BCELoss()
    params = neural_network.parameters()
    optimizer = t.optim.Adam(params=params, lr=1e-4, weight_decay=1e-5)
    final_model, best_model = train_validate_model2(neural_network, train_dataloader, validation_dataloader, test_dataloader, device, optimizer, loss_function, param.epochs, batch_size=param.batch_size)
    saved_model = f"MLP_using_{param.output}_batch_{param.batch_size}_epochs_{param.epochs}"

    # at the end we save the model and metrics which are later loaded in the "explore.ipynb" file
    t.save(final_model, f"{param.output_dir}/{saved_model}.pth")
    print(f"Saved model in {param.output_dir}/{saved_model}.pth")
    t.save(best_model, f"{param.output_dir}/best_model_{saved_model}.pth")
    print(f"Saved best model at epoch {best_model['epoch']} with loss {best_model['lowest_loss']}")

if __name__ == "__main__":
    main()