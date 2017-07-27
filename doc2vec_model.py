import json
import time
import os
import glob
from os import listdir
from os.path import isfile, join
import collections
import smart_open
import random

import numpy as np
import gensim
from gensim import corpora, models
import pandas as pd

import clean_text

def read_corpus(fname, tokens_only=False):
    with open(fname) as f:
        json_article = pd.read_json(f, typ='series')
        for i, item in enumerate(json_article):
            if tokens_only:
                yield gensim.utils.simple_preprocess(item)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(item), [i])
    return


def doc2vec_train_corpus(sourcedir):
    """Load bag-of-words dictionary and corpus from
        source bow corpus and create tagged doc2vec model"""

    start_time = time.time()

    corpus = os.path.join("./data/", sourcedir, "_doc2vec_corpus.cor")

    train_corpus = list(read_corpus(corpus))

    print("Created training corpus, took {}".format(time.time() - start_time))
    print("\n")

    return train_corpus


def doc2vec_test_corpus(sourcedir):
    """Load bag-of-words dictionary and corpus from
            source bow corpus and create untagged doc2vec model"""

    start_time = time.time()

    corpus = os.path.join("./data/", sourcedir, "_doc2vec_corpus.cor")

    test_corpus = list(read_corpus(corpus, tokens_only=True))

    print("Created testing corpus, took {}".format(time.time() - start_time))
    print("\n")

    return test_corpus


def create_doc2vec_corpus(sourcedir):

    start_time = time.time()

    texts = []

    # use glob library to open the json files in dir one by one
    # pull out the article body content and add it to the documents list
    for file in glob.iglob(os.path.join("./data/", sourcedir, "*.json")):
        with open(file) as f:
            json_article = pd.read_json(f, typ="series")
            content = json_article["content"]
            # NB: tried preprocessing using clean.text (next line), but gensim
            # preprocess in read_corpus function was more effective
            # text = clean_text.clean_text(content)

            # Add the article to the list of all articles
            texts.append(content)
            # filenumber += 1

    # save the list as the doc2vec corpus
    with open((os.path.join("./data/", sourcedir, "_doc2vec_corpus.cor")), "w") as f:
        json.dump(texts, f, ensure_ascii=False)

    print("Created doc2vec corpus, took {}".format(time.time() - start_time))
    print("\n")

    return


def train_doc2vec_model(train_corpus):

    start_time = time.time()

    # instantiate a Doc2Vec model with a vector size of 50 words,
    # iterated over the training corpus 10 times. We've set the
    # minimum word count to 2 in order to give higher frequency
    # words more weighting. Model accuracy can be improved by
    # increasing the number of iterations but this generally
    # increases the training time.

    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=10)

    # build a vocabulary i.e. a dictionary of all the unique words
    # extracted from the training corpus, along with a frequency count of those words.
    # The vocabulary is subsequently accessible via model.vocab.
    model.build_vocab(train_corpus)

    print("Created vocabulary, took {}".format(time.time() - start_time))
    print("\n")

    # train the model (Note to future self: install a BLAS library if you run this at
    # scale and want to keep training times to a minimum)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

    print("Trained model, took {}".format(time.time() - start_time))
    print("\n")

    return model


def infer_vector(model, wordlist):

    inferred_vector = model.infer_vector(wordlist)

    return inferred_vector


def assess_model(model, corpus):

    ranks = []
    second_ranks = []
    for doc_id in range(len(corpus)):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    return ranks, second_ranks


def compare_document(model, doc_id, test_corpus, train_corpus):

    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))

    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # Compare and print the most/second-most/median/least similar documents from the training corpus
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

    return


