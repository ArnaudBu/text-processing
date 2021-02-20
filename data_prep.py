#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library

import pandas as pd
from tqdm import tqdm
import zipfile
import os

# Cleaning function
from classif.classif import cl

# Unzip and load reuters files

with zipfile.ZipFile("reuters.zip", 'r') as zip_ref:
    zip_ref.extractall("reuters")

data = pd.read_csv("reuters/cats.txt", sep="|", header=None)
data[['file', 'cat']] = data[0].str.split(" ", 1, expand=True)
data[['set', 'number']] = data["file"].str.split("/", 1, expand=True)

# Process data

df = pd.DataFrame()
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    with open("reuters/" + row['file']) as f:
        txt = f.read()
    txt_raw = txt
    txt = cl(txt)
    for label in row['cat'].lower().split(" "):
        label = "__label__" + label
        set = row['set']
        df = df.append(pd.DataFrame({"txt": [txt],
                                     "txt_raw": [txt_raw],
                                     "label": [label],
                                     "set": [set]}
                                    )
                       )

data = df[['label', 'txt']]

# Write raw data
df.to_csv("data.csv", index=False)

# Write data for classification model

outdir = 'classif/data'
if not os.path.exists(outdir):
    os.mkdir(outdir)

data[df.set == "training"].to_csv(os.path.join(outdir, "train.txt"),
                                  sep='\t',
                                  index=False,
                                  header=False
                                  )
data[df.set == "test"].to_csv(os.path.join(outdir, "test.txt"),
                              sep='\t',
                              index=False,
                              header=False
                              )
data.to_csv(os.path.join(outdir, "data.txt"), sep='\t',
            index=False, header=False
            )

# Write data for word representation

outdir = 'similarity/data'
if not os.path.exists(outdir):
    os.mkdir(outdir)

data[['txt']].to_csv(os.path.join(outdir, "data.txt"), sep='\t',
                     index=False,
                     header=False
                     )