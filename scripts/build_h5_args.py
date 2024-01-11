'''
Original by: Danill Lepikhov
'''

import argparse

arg_parser = argparse.ArgumentParser(description="""
    Build the hdf5 database of torchdrug graphs from the --pdb-source into --h5out
    using --db1 as the filter (optional)
""")
# arg_parser.add_argument("--db1",
#     help="Path to db1.",
#     default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/data/BA_pMHCI_human_quantitative_HLA_A2_missing_graph.csv",
# )
# arg_parser.add_argument("--pdb",
#     help="Path to PDB files in glob format. Provide as a string.",
#     default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/data/pdb",
# )
# arg_parser.add_argument("--h5out",
#     help="Path where graphs are saved.",
#     default="/mnt/netcache/data/dlepikhov/gearnet_with_BA/data/missing_hdf5.hdf5",
# )
arg_parser.add_argument("--nworkers",
    help="Number of processes building graphs in parallel."                        
)

a = arg_parser.parse_args()