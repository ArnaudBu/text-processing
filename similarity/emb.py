#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fasttext
import os
from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Disable fasttext warnnig
fasttext.FastText.eprint = lambda x: None

# Model location
model_loc = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), "emb.bin"))


# Train the model with autotune
def train():
    model = fasttext.train_unsupervised('data/data.txt', epoch=100, dim=300)
    model.save_model("emb.bin")


class WordRep:

    def __init__(self, mdl_path=model_loc):
        """
        Word Representation utility

        Attributes
        ----------
        model : Fasttext model object
            Fasttext model object used for representation

        Parameters
        ----------
        mdl_path : str
            Path to model
        """
        self.model = fasttext.load_model(mdl_path)

    def nearest(self, n, k=1):
        """
        Returns nearest neighbors

        Parameters
        ----------
        n : str
            Names to classify

        k : int
            Number of class to returns

        Returns
        -------
        Nearest neighbors
        """
        d = self.model.get_nearest_neighbors(n, k)
        return d

    def rep(self, n):
        """
        Returns representation

        Parameters
        ----------
        n : str
            Names to classify

        Returns
        -------
        Representation
        """
        d = self.model.get_word_vector(n)
        return d

    def _txt2vec(self, sentence):
        """
        Embed sentence with model

        Parameters
        ----------
        sentence: str
                  sentence to embed

        Returns
        -------
        Sentence embedding as numpy array
        """
        emb = np.zeros(self.model.get_dimension())
        words = self.model.get_words()
        valid_words = 0
        for word in word_tokenize(sentence):
            if word in words:
                valid_words += 1
                emb += self.rep(word)
            else:
                continue
        if valid_words > 0:
            return emb / valid_words
        else:
            return emb

    def encode(self, sentences, show_progress_bar=True):
        """
        encode sentences with model

        Parameters
        ----------
        sentences: list of str
                   sentence to embed

        Returns
        -------
        Sentences embedding as numpy array
        """
        embeddings = []
        if show_progress_bar:
            for sentence in tqdm(sentences):
                embeddings.append(self._txt2vec(sentence))
        else:
            for sentence in sentences:
                embeddings.append(self._txt2vec(sentence))
        return np.array(embeddings)


class SemSearch:

    def __init__(self, sentences, mdl=WordRep()):
        """
        Semantic Search enginie

        Attributes
        ----------
        model : Model object
                Either fasttext or sentence-bert

        sentences: list of str
                   Sentences that are the search engine

        embeddings: numpy array
                    Sentences embeddings

        Parameters
        ----------
        mdl : Model object
              Either fasttext or sentence-bert

        sentences: list of str
                   Sentences that are the search engine
        """
        if not ("WordRep" in str(mdl.__class__) or "SentenceTransformer" in str(mdl.__class__)):
            raise Exception('Invalid model type')
        self.model = mdl
        self.sentences = sentences
        self.embeddings = self.model.encode(sentences, show_progress_bar=True)

    def search(self, entry, k=10):
        """
        Returns top k search results for entry

        Parameters
        ----------
        entry: str
               entry to find

        k : int
            Number of results to return

        Returns
        -------
        Search results
        """
        term_emb = self.model.encode([entry], show_progress_bar=False)
        cs = cosine_similarity(term_emb, self.embeddings)[0]
        res = pd.DataFrame({"sentence": self.sentences, "cosine": cs})
        res['entry'] = entry
        res = res.sort_values("cosine", ascending=False).iloc[0:k]
        return res.reset_index(drop=True)

    def explain(self, entry, sentence, threshold=0.4):
        """
        Explain results

        Parameters
        ----------
        entry: str
               entry to explain

        sentence: str
               sentence to explain from

        threshold : dec
            cosine distance threshold for a word

        Returns
        -------
        explained results
        """
        if not ("WordRep" in str(self.model.__class__)):
            raise Exception('Explainability only available for WordRep')
        emb = []
        emb_term = []
        words = word_tokenize(sentence)
        for word in words:
            emb.append(self.model._txt2vec(word))
        for word in word_tokenize(entry):
            emb_term.append(self.model._txt2vec(word))
        cs = cosine_similarity(emb_term, emb)
        return([words[i] for i in np.argwhere(cs > threshold)[:, 1]])

    def explain_search(self, res, threshold=0.4):
        """
        Explain results

        Parameters
        ----------
        entry: res
               output of search

        threshold : dec
            cosine distance threshold for a word

        Returns
        -------
        explained results
        """
        tqdm.pandas()
        res['explain'] = res.progress_apply(lambda x: ", ".join(self.explain(x['entry'], x['sentence'], threshold)), axis=1)
        return res


if __name__ == "__main__":
    train()
