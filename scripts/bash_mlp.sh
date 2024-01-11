#!/bin/bash
srun \
     --nodes=1\
     --ntasks=1\
     --gpus-per-task=1\
     --gpus=1\
     --cpus-per-task=16\
     --mem=20G\
     --time=1-00:00:00\
     --nodelist=dlc-articuno\
     --container-image="doduo1.umcn.nl#dlepikhov/gearnet"\
     --container-mounts=/data/cmbi/:/mnt/netcache/data\
     --pty bash