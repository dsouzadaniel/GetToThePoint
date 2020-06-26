import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))

# Project Imports
from data import loader
from utils import constant

import spacy

nlp = spacy.load('en_core_web_sm')

from sklearn.feature_extraction.text import CountVectorizer


def spacy_tokenizer(doc):
    return [token.text for token in nlp(doc)]

def tokenize_en(text: str, by_sent=True):
    doc = nlp(text)
    if by_sent:
        doc_tokens = [[token.text for token in sent] for sent in doc.sents]
    else:
        doc_tokens = [token.text for token in doc]
    return doc_tokens


def get_n_most_common_words(texts, n=50):
    vec = CountVectorizer(max_features=n, tokenizer=spacy_tokenizer)
    vec.fit(texts)
    return vec.get_feature_names()

dataset = loader.CNNLoader(path_to_csv=os.path.join(constant.DATASET_FOLDER, constant.TRAIN_FILE))
texts = [dataset[i][0] for i in range(len(dataset))]

print(get_n_most_common_words(texts, n=1024))
