import pandas as pd
import sys
import os
import torch as t
import pickle
import glob
# sys.path.append(os.path.abspath("."))
from custom_dataset import ProteinDataset
from build_h5_args import a

# gathering data and labels:
df = pd.read_csv(a.db1)
ids = df.ID.tolist()

# From the paths take the ID
pdb_paths = [p for p in pdb_paths if p.split("/")[-3] in ids]
df["binder"] = df.measurement_value.apply(lambda x: x < 500)
ids = [i.split("/")[-3] for i in pdb_paths]
df = df.loc[df.ID.isin(ids)]

# Take the binding value
targets = t.rand(len(pdb_paths), 2, dtype=t.float)
targets[df.binder.values] = t.tensor([0,1], dtype=t.float)
targets[~df.binder.values] = t.tensor([1,0], dtype=t.float)

# build graphs:
print("Generating graphs and saving them:")
dataset = ProteinDataset(
    build_h5=True,
    verbose=True,
    pdb_paths=pdb_paths,
    targets=targets,
    ntasks=10
)

dataset.save_h5(a.h5out)