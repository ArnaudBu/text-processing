# Basic text processing features for NLP

This project aims at compiling some useful text processing functions in order to easily kickstart NLP projects.


## Initialization

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data example with the Reuters-21578 benchmark corpus, ApteMod version

### Extract Data

```shell
python data_prep.py
```

### Train classification model

```shell
cd classif
python classif.py
```

### Train word representation model

```shell
cd similarity
python emb.py
```

## Text classification use case

```python
txt = "WESTERN MINING TO OPEN NEW GOLD MINE IN AUSTRALIA Western Mining Corp Holdings Ltd &lt;WMNG.S> (WMC) said it will establish a new joint venture gold mine in the Northern Territory at a cost of about 21 mln dlrs. The mine, to be known as the Goodall project, will be owned 60 pct by WMC and 40 pct by a local W.R. Grace and Co &lt;GRA> unit. It is located 30 kms east of the Adelaide River at Mt. Bundey, WMC said in a statement. It said the open-pit mine, with a conventional leach treatment plant, is expected to produce about 50,000 ounces of gold in its first year of production from mid-1988. Annual ore capacity will be about 750,000 tonnes."
```

### Detect language

```python
from language import Language
pred = Language().predict(txt, k=5) # Get 5 first guesses for language
print(list(pred[txt].keys())[0]) # Get most probable language
```

### First cleaning

> First processing by removing special characters and converting to lowercase

```python
from utils import cln
txt = cln(txt)
```

### Remove digits

> Remove words that contains at least one digit

```python
from utils import rmv_digits
txt = rmv_digits(txt)
```

### Remove stopwords

> Remove stopwords and specified words from string

```python
from utils import rmv_stp
txt = rmv_stp(txt) # default english stopwords
txt = rmv_stp(txt, stp=["stop", "words"]) # custom list
```

### Remove small words

> Remove words with too few characters

```python
from utils import rmv_smol_wds
txt = rmv_smol_wds(txt, n=1) # Remove words that are less than 1 character
```

### Lemmatize

> Lemmatize words with spaCy engine

```python
from utils import lemmatize
txt = lemmatize(txt)
```

### Classify

> Classify with the model trained with [the classification model from previous training](#train-classification-model) by default. It is also possible to specify an other model.

```python
from classif import Classr
c = Classr() # with default model
c = Classr("classif/classif.ftz") # with model path
c.predict(txt, k=5) # with string
c.predict([txt, "silver mine"]) # with list
```

## Word representation

> Word representation with fasttext. By default [the classification model from previous training](#train-word-representation-model) is used. It is also possible to specify an other model.

```python
from similarity import WordRep
w = WordRep() # with trained model
w.nearest("gold", k=3) # 3 nearest words from gold

import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # Get english model
w = WordRep('cc.en.300.bin') # Load from new model
w.nearest("gold", k=3) # 3 nearest words from gold
w.encode("gold mine") # possibility to encode a complete sentence
```

## String similarity with tf-idf

> Creation of a string matching tool based on tf-idf.

```python
from similarity import vctrz, match

sentences = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

a,b,c = vctrz(sentences) # create vector representation

d = match(sentences, b, c, sentences, t=2) # match sentences with 2 closest one
d[d['sim_cosine'] < 0.99]
```

## Semantic search

> Possibility to index documents to perform semantic search

```python
from similarity import WordRep, SemSearch # Import functions
import pandas as pd # pandas to load data

w = WordRep("similarity/emb.bin") # declare model for embeddings

data = pd.read_csv("data.csv") # import data

s = SemSearch(data['txt'], mdl=w) # declare semantic search engine with the model

res = s.search("gold mine", k=100) # perform a search with 100 results
res
s.explain("gold mine", res['sentence'][0], threshold=0.4)
# explain the sentence against the terms by finding words whose cosine
# similarity is higher than the threshold
s.explain_search(res) # apply the prior function over all the results
```

> It is also possible to use pre trained sentence embeddings from the sentence transformers package. However, it is not possible to use the explainability feature.

```python
from similarity import SemSearch # Import functions
import pandas as pd # pandas to load data

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-mean-tokens') # declare model for embeddings

data = pd.read_csv("data.csv") # import data

s = SemSearch(data['txt'], mdl=w) # declare semantic search engine with the model

res = s.search("gold mine", k=100) # perform a search with 100 results
res
```