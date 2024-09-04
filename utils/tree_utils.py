import json 
import spacy 
import numpy as np 
from tqdm import tqdm 

def get_nouns(nlp, cap):
    doc = nlp(cap)
    nouns = [] 
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            nouns.append(token.lemma_)
    return nouns

def get_verbs(nlp, cap):
    doc = nlp(cap)
    verbs = [] 
    for token in doc:
        if token.pos_ in ["VERB"]:
            verbs.append(token.lemma_)
    return verbs 

def get_nouns_verbs(nlp, cap):
    doc = nlp(cap)
    nouns = [] 
    verbs = [] 
    for token in doc:
        if token.pos_ in ["VERB"]:
            verbs.append(token.lemma_)
        elif token.pos_ in ["NOUN", "PROPN"]:
            nouns.append(token.lemma_)
    return nouns, verbs 

def extract_nouns(nlp, vocab, sentence):
    nouns = get_nouns(nlp, sentence)
    nouns = list(filter(lambda x: x in vocab, nouns))
    return nouns 

def extract_verbs(nlp, vocab, sentence):
    verbs = get_verbs(nlp, sentence)
    verbs = list(filter(lambda x: x in vocab, verbs))
    return verbs 

def extract_keys(nlp, vocab, sentence, max_cnt=50):
    nouns, verbs = get_nouns_verbs(nlp, sentence)
    keys = nouns + verbs
    keys = list(filter(lambda x: x in vocab, keys))
    keys = keys[:max_cnt]
    return keys 