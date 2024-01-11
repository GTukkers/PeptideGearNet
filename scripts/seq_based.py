import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from finetune_args import param
import time
import warnings

# Assuming the warning message is triggered here
def ignore_warning():
    return warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")


def set_device():
    device = (torch.device("cpu"), torch.device("cuda:0"))[torch.cuda.is_available()]
    print(f"Device used: {device}")
    return device

# Function to load and preprocess dataframe
def load_and_preprocess_dataframe(path):
    df = pd.read_csv(path)
    df = df.loc[(df.peptide.str.len() == 9) & (df.allele == "HLA-A*02:01")]
    df["binder"] = df.measurement_value.apply(lambda x: int(x < 500))
    df["one_hot"] = df.peptide.apply(lambda x: protein_to_onehot(x))
    df['one_hot'] = df['one_hot'].apply(lambda x: np.array(x))
    return df

def split_data(df):
    train_df, validation_df = train_test_split(df, test_size=0.2, stratify=df.binder, random_state=1)
    print(f"len of train_ids: {len(train_df)}\nlen of validation_ids: {len(validation_df)}")
    return train_df, validation_df

#To create the onehot encoding for the amino acid
def protein_to_onehot(seq):
    # Make an Identity matrix with 20 rows as to encode the 20 AAs
    AAs_hot = torch.eye(20, dtype=torch.float)
    aminoacids = ('ACDEFGHIKLMNPQRSTVWY')
    return torch.stack([AAs_hot[aminoacids.index(aa)]for aa in seq])

def load_dataset(df):
   target = torch.tensor(df.binder.values.tolist())
   one_hot = torch.tensor(df.one_hot.values.tolist())
   return TensorDataset(one_hot, target)

def create_data_loader(dataset, batch_size, n_workers):
   return DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True, drop_last=True)


# Define your training function
def train_epoch(neuralnet, train_dataloader, device, optimizer, loss_function, train_metrics, input_size, train_gearnet_weights=None, batch_size=None, dataframe=None):
    neuralnet.train()

    for (minib, batch) in enumerate(train_dataloader):
        target = batch[1].float().unsqueeze(-1).to(device)
        # The last batch is not the rigth size
        onehot = batch[0].reshape(16, 9, 20).to(device)
        predict = neuralnet(onehot)
        #print(f"Target shape: {target.shape}")
        #print(f"Predict shape: {predict.shape}")
        peptide_loss = loss_function(predict, target)

      #Updating neural network
        optimizer.zero_grad()
        peptide_loss.backward()
        optimizer.step()

        if minib % 10 == 0:
            print(f"Training loss at minibatch {minib}: {peptide_loss:.5f}")
        
        #Save the following metrics
        '''
        What do I care about the losses and everything per bach
        If I were to change it to, per epoch also easier to retrieve information
        store in list, add mean to dictionary 
        '''
        train_metrics["losses"].append(peptide_loss)
        train_metrics["predict"].append(predict)
        train_metrics["targets"].append(target)


# Define your validation function
def validate_epoch(neuralnet, validation_dataloader, device, loss_function, val_metrics, input_size, batch_size=None, dataframe=None):
    neuralnet.eval()
    epoch_val_loss = []

    for (minib, batch) in enumerate(validation_dataloader):
        target = batch[1].float().unsqueeze(-1).to(device)
        onehot = batch[0].reshape(16, 9, 20).to(device)
        predict = neuralnet(onehot)
        peptide_loss = loss_function(predict, target)
        
        if minib % 10 == 0:
            print(f"Validation loss at minibatch {minib}: {peptide_loss:.5f}")
        
        #Why do I need the following 
        epoch_val_loss.append(peptide_loss)
        #Save the following metrics for each batch
        val_metrics["losses"].append(peptide_loss)
        val_metrics["predict"].append(predict)
        val_metrics["targets"].append(target)

    epoch_val_loss = torch.stack(epoch_val_loss)
    #Need to return the loss in order to know which model to save
    return epoch_val_loss

def train_model(neural_network, train_dataloader, validation_dataloader, test_dataloader, device, optimizer, loss_function, epochs, batch_size=None):
    best_model = {
        "model": None,
        "lowest_loss": 10000,
        "epoch": None,
        "test_loss" : None

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
        train_epoch(neural_network, train_dataloader, device, optimizer, loss_function, metrics["train"], batch_size)
        epoch_valid_loss = validate_epoch(neural_network, validation_dataloader, device, loss_function, metrics["val"], batch_size)

        #Save the best model, this is based on the lowest loss of the validation set
        if epoch_valid_loss.mean() < best_model["lowest_loss"]:
            best_model["model"] = neural_network.state_dict()
            best_model["lowest_loss"] = epoch_valid_loss.mean()
            best_model["epoch"] = epoch

        t2 = time.time()
        print(f"Time for epoch (train and evaluation): {t2-t1} seconds")
        metrics["all_times"].append(t2-t1)

    final_model["metrics"] = metrics
    final_model["model"] = neural_network.state_dict()

    #Reload the information of the best neural network, to run the test set
    neural_network.load_state_dict(best_model["model"])
    test_loss_best = validate_epoch(neural_network, test_dataloader, device, loss_function, metrics["test"], batch_size)
    best_model["test_loss"] = test_loss_best.mean()
    return final_model, best_model


def main():
    ignore_warning()
    device = set_device()
    df = load_and_preprocess_dataframe(param.db1_path)
    test_df = load_and_preprocess_dataframe(param.db1_test)
    
    merged = pd.concat([df, test_df], ignore_index=True)

    # train_df, validation_df = split_data(df)
    print(f"len of test_id: {len(test_df)}")

    df = pd.read_csv(param.OOD)
    train_ids = df.loc[df['datatype'] == "train", 'ID'].tolist()
    validation_ids = df.loc[df['datatype'] == "val", 'ID'].tolist()
    test_ids = df.loc[df['datatype'] == "test", 'ID'].tolist()

    train_df = merged[merged["ID"].isin(train_ids)]
    validation_df = merged[merged["ID"].isin(validation_ids)]
    test_df = merged[merged["ID"].isin(test_ids)]

    train_dataset = load_dataset(train_df)
    validation_dataset = load_dataset(validation_df)
    test_dataset = load_dataset(test_df)

    train_dataloader = create_data_loader(train_dataset, param.batch_size, param.num_workers)
    validation_dataloader = create_data_loader(validation_dataset, param.batch_size, param.num_workers)
    test_dataloader = create_data_loader(test_dataset, param.batch_size, param.num_workers)

    INPUT_NEURONS = 9*20
    OUTPUT_NEURONS = 1
    HIDDEN_NEURONS = 320

    neural_network = nn.Sequential(nn.Flatten(),
                                nn.Linear(INPUT_NEURONS, HIDDEN_NEURONS),
                                nn.BatchNorm1d(HIDDEN_NEURONS),
                                nn.ReLU(),
                                nn.Linear(HIDDEN_NEURONS, OUTPUT_NEURONS),
                                nn.Sigmoid())

    # Optimizer keeps track of gradients
    neural_network.to(device)
    optimizer = torch.optim.Adam(neural_network.parameters())

    loss_function = torch.nn.BCELoss()

    saved_model = f"MLP_using_{param.output}_batch_{param.batch_size}_epochs_{param.epochs}"
    final_model, best_model = train_model(neural_network, train_dataloader, validation_dataloader, test_dataloader, device, optimizer, loss_function, param.epochs, batch_size=param.batch_size)
    torch.save(final_model, f"{param.output_dir}/{saved_model}.pth")
    print(f"Saved model in {param.output_dir}/{saved_model}.pth")
    torch.save(best_model, f"{param.output_dir}/best_model_{saved_model}.pth")
    print(f"Saved best model at epoch {best_model['epoch']} with loss {best_model['lowest_loss']}")

   
if __name__ == "__main__":
  main()