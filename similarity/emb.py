#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fasttext
import os

# Disable fasttext warnnig
fasttext.FastText.eprint = lambda x: None

# Model location
model_loc = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), "emb.bin"))


# Train the model with autotune
def train():
    model = fasttext.train_unsupervised('data/data.txt', epoch=100)
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


if __name__ == "__main__":
    train()
