#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fasttext
import re
import os
from unidecode import unidecode

# Disable fasttext warnnig
fasttext.FastText.eprint = lambda x: None

# Model location
model_loc = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), "lid.176.ftz"))


# Class for language processing
class Language:

    def __init__(self):
        """
        Classifier used for language.

        """
        self.model = fasttext.load_model(model_loc)

    def predict(self, n, k=1):
        """
        Predictor for the language

        Parameters
        ----------
        n : str or list of str
            String to classify

        k : int
            Number of class to returns

        Returns
        -------
        dictionnary
            Dictionnary with languages and their scores
        """
        if type(n) is not list:
            n = [n]
        r = self.model.predict([a for a in n], k=k)
        d = dict()
        for i in range(len(n)):
            l = dict()
            for j in range(len(r[0][i])):
                l[re.sub("__label__", "", r[0][i][j])] = r[1][i][j]
            d[n[i]] = l
        return d
