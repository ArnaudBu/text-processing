#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Packages
import logging
import sys
import functools
import re
from unidecode import unidecode
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

stp = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def create_logger(logger_name, type="stream", file="error.log"):
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    fmt = '%(pathname)s:%(lineno)d : %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    if type == "stream":
        fh = logging.StreamHandler(sys.stdout)
    else:
        fh = logging.FileHandler(file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def exception(function):
    """
    A decorator that wraps the passed in function and logs exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logger = create_logger("error_log")
        try:
            return function(*args, **kwargs)
        except:
            e = sys.exc_info()[0]
            logger.exception(e)
            # re-raise the exception
            # raise
    return wrapper

def cln(s, rm=r'!"#$%&\'()*+-./:;<=>?@[\]^_`{|}~'):
    """
    Clean string and remove some characters
    """
    return re.sub(' +', ' ',
                  unidecode(s.lower()).
                  translate(str.maketrans(rm, ' '*len(rm))).
                  strip().
                  replace('\n', ' '))

def rmv_stp(s):
    """
    Clean string and remove some characters
    """
    return ' '.join([b for b in s.split(" ") if b not in stp])

def rmv_digits(s):
    """
    Remove words with digits from string
    """
    return ' '.join(b for b in s.split(" ") if not any(c.isdigit() for c in b))

def rmv_smol_wds(s, n=1):
    """
    Remove words with less than n characters
    """
    return ' '.join([b for b in s.split(" ") if len(b) > n])

def lemmatize(s):
    """
    Lemmatize a string
    """
    doc = nlp(s)
    return " ".join([token.lemma_ for token in doc])