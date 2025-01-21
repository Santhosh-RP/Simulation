#!/usr/bin/env python3

import os
import sys
import itertools
import json
import time
import random

with open('done.json') as f:
    done = json.load(f)
with open('tasks.json') as f:
    tasks = json.load(f)

for task in tasks:
    id = task[0]
    if id in done:
        print(f"Skipping {id}")
        continue

    cfg = task[1]
    ttype, batch_size, learning_rate, num_epochs, max_patience, zdim, vit_num_classes, vit_mlp_dim, vit_hidden_dim, vit_num_heads, vit_num_layers = cfg
    vit_hidden_dim *= vit_num_heads

    if ttype == "cnn5":
        params = f"{batch_size} {learning_rate} {num_epochs} {max_patience} {zdim}"
        call = f"python3 vae_train.py {id} {params}"
    elif ttype == "vit10":
        params = f"{batch_size} {learning_rate} {num_epochs} {max_patience} {zdim} {vit_num_classes} {vit_mlp_dim} {vit_hidden_dim} {vit_num_heads} {vit_num_layers}"
        call = f"python3 vitvae_train.py {id} {params}"
        ret = os.system(call)
    else:
        print(f"WTF {ttype=}")

    print(call)
    ret = os.system(call)
    print('task done')

    if ret == 0:
        done.append(id)
        with open('done.json', "w") as f:
            json.dump(done, f, indent=2)
    print(task)
