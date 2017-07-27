import argparse
import os
import time

from gensim import corpora, models


def create_tfidf_corpus(sourcedir):
    """Load bag-of-words dictionary and corpus from
    source folder of classified jsons and create tfidf model"""

    start_time = time.time()

    corpus = corpora.MmCorpus(os.path.join("./data/Topics/", sourcedir, "_bow_corpus.mm"))

    # Initialize a tfidf model
    tfidf = models.TfidfModel(corpus)
    tfidf.save(os.path.join("./data/Topics/", sourcedir, "_tfidf_model.mm"))

    print("Finished corpus and dictionary creation, took {}".format(time.time() - start_time))
    print("\n")

    return tfidf


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

    # Call the function to create the  dictionary and
    # corpus from the data in the source directory
    create_tfidf_corpus(sourcedir)

