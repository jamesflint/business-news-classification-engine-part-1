import os

from gensim import corpora, models
from nltk.stem import WordNetLemmatizer

import clean_text

wnl = WordNetLemmatizer()


def bow_vectorize(sourcedir, content):
    """Perform a bow transform on submitted content"""

    # Clean & lemmatize the raw training content
    text = clean_text.clean_text(content)
    # print("\nCleaned & lemmatized training text:\n\n", text)

    dictionary = corpora.Dictionary.load(os.path.join("./data/Topics/", sourcedir, "_bow_corpus.dict"))

    # Vectorize the content using dictionary created from clean content
    vector = dictionary.doc2bow(text)  # (content.lower().split())
    # print("\nVector of training text:", vector)

    return vector


def tfidf_vectorize(sourcedir, content):
    """Perform a tfidf transform on submitted content"""

    # Clean & lemmatize the raw training content
    text = clean_text.clean_text(content)
    # print("\nCleaned & lemmatized training text:\n\n", text)

    dictionary = corpora.Dictionary.load(os.path.join("./data/Topics/", sourcedir, "_bow_corpus.dict"))

    # Vectorize the content using dictionary created from clean content
    vector = dictionary.doc2bow(text)  # (content.lower().split())
    # print("\nVector of training text:", vector)

    #tfidf = tfidf.load(os.path.join("./data/Topics/", sourcedir, "_tfidf_model.mm"))
    tfidf = models.tfidfmodel.TfidfModel.load(os.path.join("./data/Topics/", sourcedir, "_tfidf_model.mm"))

    # Perform tfidf transformation on the vectorized content
    vector = tfidf[vector]

    return vector
