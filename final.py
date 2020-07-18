import sys
import pickle
import os
import glob
import time
from GenerateEmbeddings import generate_embeddings_narrow, generate_embeddings_wide

# Initialize two directories
input_dir= sys.argv[1]
output_dir= sys.argv[2]

# Read the two datasets
input_path_narrow= input_dir + '/dataset-narrow'
input_path_wide= input_dir + '/dataset-wide'

# Create two subfolders within output dir, dataset-narrow and dataset-wide
output_path_narrow= output_dir + '/dataset-narrow'
output_path_wide= output_dir + '/dataset-wide'

os.mkdir(output_path_narrow)
os.mkdir(output_path_wide)

dataset_narrow= glob.glob(input_path_narrow+'/*.txt')
dataset_wide= glob.glob(input_path_wide+'/*.txt')

try:
    generate_embeddings_narrow(dataset_narrow, input_path_narrow, output_path_narrow)
    del dataset_narrow

    generate_embeddings_wide(dataset_wide, input_path_wide, output_path_wide)
    del dataset_wide

except:
    pass