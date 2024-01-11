#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --container-image="doduo1.umcn.nl#dlepikhov/gearnet"
#SBATCH --container-mounts=/data/cmbi/:/mnt/netcache/data


python3 /mnt/netcache/data/dlepikhov/gearnet_with_BA/script/one_hot_mlp.py\
    --epochs 25\
    --output-dir /mnt/netcache/data/dlepikhov/gearnet_with_BA/models_gijs\
    --batch-size 32\
    --mc-path /mnt/netcache/data/dlepikhov/gearnet_with_BA/models/mc_gearnet_edge.pth\
    --num-workers 16\
    --h5-path /mnt/netcache/data/dlepikhov/gearnet_with_BA/data/proteins.hdf5\
    --output one_hot_encoding\
    --db1 /mnt/netcache/data/dlepikhov/3DVac_experiments/experiments_data/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv
 