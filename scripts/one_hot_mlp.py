'''
Sequence based mlp on HLA-A*02:01 9mers only
Info dataset: 
    Size: 8000
    Classification: binar
'''
import resource
import torch 
from finetune_args import a 
import pandas as pd
from sklearn.model_selection import train_test_split
from custom_dataset import ProteinDataset, MLP
import torch.nn as nn
import random

def protein_to_onehot(seq):
    # Make an Identity matrix with 20 rows as to encode the 20 AAs 
    AAs_hot = torch.eye(20, dtype=torch.float)

    aminoacids = ('ACDEFGHIKLMNPQRSTVWY')
    return torch.stack([AAs_hot[aminoacids.index(aa)]for aa in seq])

## Apply the encoding to all proteins in any given protein list
def proteins_to_onehot(proteins):
  encoded = torch.stack([protein_to_onehot(prot) for prot in proteins])
  return encoded

# Lines of code to to fix the "RuntimeError: received 0 items of ancdata" which happens sometimes on deepops cluster:
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# Check that CUDA is being used:
device = (torch.device("cpu"),torch.device("cuda:0"))[torch.cuda.is_available()]
print(f"Device used: {device}")


# Database containing the caseID, allele, peptide and label column
hla_df = pd.read_csv(a.db1)
# We work only on 9-mer peptides from the HLA-A*02:01 allele
hla_df = hla_df.loc[(hla_df.peptide.str.len() == 9) & (hla_df.allele == "HLA-A*02:01")]
ids = hla_df.ID.tolist()

#Getting the ids which are used to define the training set and the validation set for the neural network
train_ids, validation_ids = train_test_split(ids, test_size= .2, stratify= hla_df.binder, random_state= 1)

print(f"length dataset: {len(hla_df)}")
print(f"length train: {len(train_ids)}")
print(f"length validation: {len(validation_ids)}")

# hla_df["binder"] is used as the label (already loaded in the hhla_df5 file)
hla_df["binder"] = hla_df.measurement_value.apply(lambda x: int(x < 500))

peptides = hla_df.peptide
labels = torch.tensor(hla_df.binder.values)

# Creating the one-hot embedding for the neural network
encoded_peptides = proteins_to_onehot(peptides)

train_data = encoded_peptides[train_ids]
train_labels = labels[train_ids]
validation_data = encoded_peptides[validation_ids]
validation_labels = labels[validation_ids]


#Setting up the neural network
INPUT_NEURONS = 9*20
OUTPUT_NEURONS = 1 
HIDDEN_NEURONS = 10
EPOCH_NUMBER = 50


neural_network = nn.Sequential(nn.Flatten(),
                          nn.Linear(INPUT_NEURONS, HIDDEN_NEURONS),
                          nn.BatchNorm1d(HIDDEN_NEURONS),
                          nn.ReLU(),
                          nn.Linear(HIDDEN_NEURONS, OUTPUT_NEURONS),
                          nn.Sigmoid())

optimizer = torch.optim.Adam(neural_network.parameters())
#Start of training training the neural network over epochs

all_losses = []
all_e_losses = []
all_targets = []
all_e_targets = []
all_pred = []
all_e_pred = []
all_times = []

for epoch in range(EPOCH_NUMBER):
  predict_binding = neuralnet(encoded_peptides)
  loss = neuralnet.eval()


# Train one epoch
import matplotlib.pyplot as plt

def train_one_epoch(dataset, labels, neuralnetwork, optimizer):
  neuralnetwork.train()
  # Shuffle your dataset
  shuffle = torch.randperm(len(dataset))
  dataset = dataset[shuffle]
  labels  =  labels[shuffle]

  prediction = neural_network(dataset) # Get the neural network prediction
  
  error = torch.square(labels - prediction)  # Calculate mean squared error
  error = torch.mean(error)

  # Use gradient descent to update the neural network
  optimizer.zero_grad() # clear old gradients
  error.backward() # -> calcultate new gradients
  optimizer.step() # -> update the model's weight using new gradients
  # add the error to the total error
  loss = error.item()
  # A variable that is used to keep up the error of the neural network
  return loss

# Train the network
train_accuracies = []
val_accuracies = []
losses = []
def train_nn(train_data,val_data, train_labels,val_labels, neuralnetwork, optimizer, epochs):
  for epoch in range(epochs):
    
    train_acc = getAccuracy(train_data, neuralnetwork, train_labels)
    val_acc   = getAccuracy(val_data  , neuralnetwork, val_labels)
    
    loss_train = train_one_epoch(train_data, train_labels, neuralnetwork, optimizer)
    loss_validation = train_one_epoch(train_data, train_labels, neuralnetwork, optimizer)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    losses.append(loss)

# Calculates accuracy
def calcAccuracy(predictions, labels):
  return (predictions.round()==labels).sum()/float(len(predictions)) * 100

# For any given dataset retruns the accuracy of the predicted labels by the network
def getAccuracy(dataset, neuralnetwork, labels):
  neuralnetwork.eval()
  with torch.no_grad():
    predictions = neuralnetwork(dataset)
    acc = calcAccuracy(predictions, labels)
    return acc

# Train the network with our training data
train_nn(train_data,val_data, train_labels,val_labels, neural_network, optimizer, EPOCH_NUMBER)


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

final_name = f"{a.output_dir}/{saved_model}.pth"

# at the end we save the model and metrics which are later loaded in the "explore.ipynb" file
t.save(final_model, final_name)
print(f"Saved model in {final_name}")














'''
The hla_df already contains the peptide sequence, therefor I can create an extra column in the hla_df which contains the one-hot encoding
'''

