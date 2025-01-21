#!/usr/bin/env python3

import os
import sys
import itertools
import json
import time
import random

# general
typeL = ['cnn5', 'vit10']
# cnn
batch_sizeL = [20, 32, 64, 128] 
learning_rateL = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
num_epochsL = [500]
max_patienceL = [50]
zdimL = [32, 64, 96, 128, 150, 256]
# vit encoder
vit_num_classesL = [32, 64, 96, 128, 150, 256, 512]
vit_mlp_dimL = [64, 96, 128, 150, 256, 512]
vit_hidden_dim_MUL_L = [4,8,12,16] #  *mul_heads
vit_num_headsL = [4,8,12,16]
vit_num_layersL = [3,4,5,6]


# Check the filename doesn't exist
output_fname = "tasks.json"
if os.path.isfile(output_fname):
    print(f"Output file {output_fname} exists. The script will exit to prevent lossing data.")
    sys.exit(-1)

params = typeL, batch_sizeL, learning_rateL, num_epochsL, max_patienceL, zdimL, vit_num_classesL, vit_mlp_dimL, vit_hidden_dim_MUL_L, vit_num_headsL, vit_num_layersL
tasks = [ [str(x).zfill(7), list(y)] for x, y in enumerate(itertools.product(*params))]
random.shuffle(tasks)

out_file = open(output_fname, "w") 
json.dump(tasks, out_file, indent=2)

