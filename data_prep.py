#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library

import pandas as pd
from tqdm import tqdm
from language import Language
from utils import cln, rmv_digits, rmv_stp, rmv_smol_wds, lemmatize
import zipfile
import os

# Cleaning function
def cl(s):
    return cln(rmv_smol_wds(rmv_digits(lemmatize(rmv_stp(s)))))

# Unzip and load reuters files

with zipfile.ZipFile("reuters.zip", 'r') as zip_ref:
    zip_ref.extractall("reuters")

data = pd.read_csv("reuters/cats.txt", sep="{-_-}", header=None)
data[['file','cat']] = data[0].str.split(" ", 1, expand=True)
data[['set','number']] = data["file"].str.split("/", 1, expand=True)

# Process data

df = pd.DataFrame()
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
       with open("reuters/" + row['file']) as f:
              txt = f.read()
       txt = cl(txt)
       label = "__label__" + row['cat'].lower().replace(" ", "_")
       set = row['set']
       df = df.append(pd.DataFrame({"txt": [txt], "label": [label], "set": [set]}))

data = df[['label', 'txt']]

# Write data for classification model

outdir = 'classif/data'
if not os.path.exists(outdir):
    os.mkdir(outdir)

data[df.set == "training"].to_csv(os.path.join(outdir, "train.txt"), sep='\t', index = False, header = False)
data[df.set == "test"].to_csv(os.path.join(outdir, "test.txt"), sep='\t', index = False, header = False)
data.to_csv(os.path.join(outdir, "data.txt"), sep='\t', index = False, header = False)

# Write data for word representation

outdir = 'similarity/data'
if not os.path.exists(outdir):
    os.mkdir(outdir)

data[['txt']].to_csv(os.path.join(outdir, "data.txt"), sep='\t', index = False, header = False)