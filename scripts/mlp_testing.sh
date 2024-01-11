#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --nodelist=dlc-groudon\
#SBATCH --container-image="doduo1.umcn.nl#dlepikhov/gearnet"
#SBATCH --container-mounts=/data/cmbi/:/mnt/netcache/data

python3 /mnt/netcache/data/dlepikhov/gearnet_with_BA/script/gearnet_embedding_one_hot.py\
    --epochs 150\
    --output-dir /mnt/netcache/data/dlepikhov/gearnet_with_BA/models_gijs\
    --batch-size 16\
    --mc-path /mnt/netcache/data/dlepikhov/gearnet_with_BA/models/mc_gearnet_edge.pth\
    --num-workers 16\
    --h5-path /mnt/netcache/data/dlepikhov/gearnet_with_BA/data/proteins.hdf5\
    --hidden1 20\
    --hidden2 320\
    --hidden3 240\
    --output 3072_Linear_NoBias_no_dropout_One_hot\
 