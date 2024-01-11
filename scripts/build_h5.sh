#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=dlc-groudon
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH --container-image="doduo1.umcn.nl#dlepikhov/gearnet"
#SBATCH --container-mounts=/data/cmbi/:/mnt/netcache/data


cd 
python3 -u /mnt/netcache/data/dlepikhov/gearnet_with_BA/script/build_h5.py \
