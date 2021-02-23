#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fasttext
import re
import os
import sys
from cleantext import clean

try:
    from utils import cln, rmv_digits, rmv_stp, rmv_smol_wds, lemmatize
except:
    sys.path.append("..")
    from utils import cln, rmv_digits, rmv_stp, rmv_smol_wds, lemmatize


# Cleaning function
def cl(s):
    # return rmv_smol_wds(rmv_digits(rmv_stp(cln(s))))
    a = clean(s,
              fix_unicode=True,
              to_ascii=True,
              lower=True,
              no_line_breaks=True,
              no_urls=True,
              no_emails=True,
              no_phone_numbers=True,
              no_numbers=True,
              no_emoji=True,
              replace_with_url=" ",
              replace_with_email=" ",
              replace_with_phone_number=" ",
              replace_with_number=" ",
              lang="en"
              )
    return rmv_smol_wds(lemmatize(rmv_stp(a)))


# Disable fasttext warnnig
fasttext.FastText.eprint = lambda x: None

# Model location
model_loc = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), "classif.ftz"))


# Print fasttext results
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.5f}".format(1, p))
    print("R@{}\t{:.5f}".format(1, r))


# Default train parameters
train_parameters = {
    'lr': 0.524748798845577,
    'dim': 15,
    'ws': 5,
    'epoch': 11,
    'minCount': 1,
    'minCountLabel': 0,
    'minn': 3,
    'maxn': 6,
    'neg': 5,
    'wordNgrams': 3,
    'bucket': 825039,
    'lrUpdateRate': 100,
    't': 0.0001,
    'verbose': 2,
    'pretrainedVectors': '',
    'seed': 0
 }


# Get parameters from model
def get_model_parameters(model):
    args_getter = model.f.getArgs()

    parameters = {}
    for param in train_parameters:
        attr = getattr(args_getter, param)
        if param == 'loss':
            attr = attr.name
        parameters[param] = attr

    return parameters


# Train the model with autotune
def train():
    model = fasttext.train_supervised(input=r"data/train.txt",
                                      autotuneValidationFile=r"data/test.txt",
                                      autotuneDuration=60*60)

    print_results(*model.test(r"data/test.txt"))
    print_results(*model.test(r"data/train.txt"))

    train_parameters = get_model_parameters(model)

    model = fasttext.train_supervised(input=r"data/data.txt",
                                      **train_parameters
                                      )

    model.save_model("classif.ftz")
    print_results(*model.test(r"data/data.txt"))
    model.quantize(input=r"data/data.txt", retrain=True)
    model.save_model("classifq.ftz")


class Classr:

    def __init__(self, mdl_path=model_loc):
        """
        Classifier

        Attributes
        ----------
        model : Fasttext model object
            Fasttext model object used for prediction

        Parameters
        ----------
        mdl_path : str
            Path to model
        """
        self.model = fasttext.load_model(mdl_path)

    def predict(self, n, k=1, cl=cl):
        """
        Classifies the input.

        Parameters
        ----------
        n : str or list of str
            Strings to classify

        k : int
            Number of class to returns

        cl: function
            Cleaning function

        Returns
        -------
        dictionnary
            Dictionnary of the input strings with a tuple as value containing
            the classification and the score associated to this classification.
        """
        if type(n) is not list:
            n = [n]
        r = self.model.predict([cl(a) for a in n], k=k)
        d = dict()
        for i in range(len(n)):
            l = dict()
            for j in range(len(r[0][i])):
                l[re.sub("__label__", "", r[0][i][j])] = r[1][i][j]
            d[n[i]] = l
        return d


if __name__ == "__main__":
    train()
