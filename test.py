from classif import Classr
from language import Language

import time


classr = Classr()

test_headline = "Adidas Struggles on Home Turf as Nike Muscles In;"
list(Language().predict(test_headline)[test_headline].keys())[0]

start = time.time()

c1 = classr.predict(test_headline, 12)


end = time.time()
print("Classif fasttext")
print(c1)
print("Time taken")
print(end - start)

import pandas as pd
from similarity import vctrz, match
df = pd.read_csv("similarity/data/data.txt", sep=";", header=None)

vec = df[0][df[0].notnull()].tolist()

a,b,c = vctrz(vec)

match(["Hello World"],b,c,vec, t=10)

from similarity import WordRep

w = WordRep()