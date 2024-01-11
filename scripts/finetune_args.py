'''
Original by: Danill Lepikhov
Parameters used in the main finetune script
'''

import argparse

arg_parser = argparse.ArgumentParser(description="""
    Finetunes the gearnet multicontrast view model on BA prediction
""")
arg_parser.add_argument("--with-gearnet-weights",
    help="Finetunes only the last MLP layers of the model. Default to False.",
    default = False,
    action = "store_true"
)
arg_parser.add_argument("--concat-gearnet",
    help = "concatenate the hidden layers of the neuralnetwork",
    default = True,
    type= bool
)
arg_parser.add_argument("--mc-path",
    help="Path to MC weights",
    default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/models/mc_gearnet_edge.pth"
)
arg_parser.add_argument("--output-dir",
    help="Directory where .pth model will be saved.",
    default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/models_gijs"
)
arg_parser.add_argument("--batch-size",
    help="Batch size. Default 32",
    type=int,
    default=16
)
arg_parser.add_argument("--epochs",
    help="Number of iterations over the training dataset. Default 50.",
    default=150,
    type=int
)
arg_parser.add_argument("--num-workers",
    help="Number of workers for dataloaders. Default 32.",
    type=int,
    default=16
)
arg_parser.add_argument("--h5-path",
    help="""Path to the hdf5 file containing graphs and targets. 
    Default /projects/0/einf2380/data/gearnet/proteins.hdf5 """,
    default = "/mnt/netcache/data/dlepikhov/gearnet_with_BA/data/proteins.hdf5"
)
arg_parser.add_argument("--output",
    help="""
    Name of the saved PyTorch file. Default `shuffled`.
    """,
    default="3072_Linear_NoBias_no_dropout_One_hot" 
)
arg_parser.add_argument("--db1-path",
    help="""
    Path to the DB1 (csv of the experiment containing train and validation cases). 
    Default: /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv
    """,
    default = "/mnt/netcache/data/dlepikhov/3DVac_experiments/experiments_data/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv"
)
arg_parser.add_argument("--db1-test",
    help="""
    Path to the DB1 (csv of the experiment containing test cases). 
    Default: /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_test.csv
    """,
    default = "/mnt/netcache/data/dlepikhov/3DVac_experiments/experiments_data/BA_pMHCI_human_quantitative_only_eq_shuffled_test.csv"
)

arg_parser.add_argument("--hidden1",
    type=int,
    default=20
)
arg_parser.add_argument("--hidden2",
    type=int,
    default=320
)
arg_parser.add_argument("--hidden3",
    type=int,
    default=64
)
arg_parser.add_argument("--dropout",
    type=int,
    default=0.2
)
arg_parser.add_argument("--OOD",
    default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/data/OOD_1_cluster_dataset_IDs_datatype.csv"
)
arg_parser.add_argument("--autoencoder",
    default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/models_gijs/best_model_MLP_using_Autoencoder_180_neurons_batch_16_epochs_25.pth"
    )
arg_parser.add_argument("--last_model",
    default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/models_gijs/MLP_using_Encoder_and_one_hidden_layer_further_batch_16_epochs_25.pth"
    )
arg_parser.add_argument("--embedding_512",
    default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/data/embedding/GearNet-Edge_embedding_512.pkl"
    )
arg_parser.add_argument("--embedding_3072",
    default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/data/embedding/GearNet-Edge_embedding_3072.pkl"
    )
arg_parser.add_argument("--loo_df",
    default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/data/LOO_dataset.csv"
    )
param = arg_parser.parse_args()