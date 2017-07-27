import argparse
import glob
import os
import time

import pandas as pd
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer

import clean_text

wnl = WordNetLemmatizer()

def create_bow_corpus(sourcedir):
    """Load body content from folder of classified jsons and create bag-of-words corpus"""

    start_time = time.time()

    # create a list 'texts' to store the article body content for creating the corpus
    texts = []
    filenumber = 1  # File counter for use when iterating files and registering file ID

    # use glob library to open the json files in dir one by one
    # pull out the article body content and add it to the documents list
    for file in glob.iglob(os.path.join("./data/Topics/", sourcedir, "*.json")):
        with open(file) as f:
            json_article = pd.read_json(f, typ="series")
            content = json_article["content"]
            text = clean_text.clean_text(content)
            # Add the article to the list of all articles
            texts.append(text)
            filenumber += 1

    # print('\nCleaned and lemmatized corpus texts:\n\n', texts)

    # Create the dictionary
    dictionary = corpora.Dictionary(texts)

    # Filter out low frequency words and compact AFTER dictionary is created (preferred method)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    dictionary.compactify()

    # Save the filtered dictionary, for future reference
    dictionary.save(os.path.join("./data/Topics/", sourcedir, "_bow_corpus.dict"))
    print("Dictionary:", dictionary, "\n")

    # Create and save the corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Save to disk, for later use
    corpora.MmCorpus.serialize((os.path.join("./data/Topics/", sourcedir, "_bow_corpus.mm")), corpus)

    print("Finished bag-of-words corpus and dictionary creation, took {}".format(time.time() - start_time))

    return corpus


if __name__ == "__main__":

    # Parser setup to create source_dir path
    parser = argparse.ArgumentParser()
    parser.prog = "SOURCE_DIR"
    parser.description = "Get the source directory for the bow corpus and dictionary."
    parser.add_argument("get_sourcedir", help="Source directory")
    # parser.add_argument("get_targetdir", help="Target directory")
    args = parser.parse_args()
    sourcedir = str(args.get_sourcedir)
    print("Source directory =", sourcedir)
    print("Corpus is stored in", sourcedir)

    # Call the function to create the  dictionary and corpus from the data in the source directory
    create_bow_corpus(sourcedir)


