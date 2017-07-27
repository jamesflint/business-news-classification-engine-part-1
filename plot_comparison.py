import argparse
import glob
import math
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora, models

import vectorize
import doc2vec_model


def plot_comparison(sourcedir, sampledir, langmodel):
    """Opens the json files in the specified sample directory one by one, sends the article content
    for vectorizing, then performs cosine similarity against the corpus in the specified source directory"""

    start_time = time.time()

    comparison_df = pd.DataFrame()

    if langmodel == "bow":
        corpus = corpora.MmCorpus(os.path.join("./data/Topics/", sourcedir, "_bow_corpus.mm"))
    elif langmodel == "tfidf":
        corpus = corpora.MmCorpus(os.path.join("./data/Topics/", sourcedir, "_bow_corpus.mm"))
        tfidf = models.tfidfmodel.TfidfModel.load(os.path.join("./data/Topics/", sourcedir, "_tfidf_model.mm"))
        # tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]
    elif langmodel == "doc2vec":
        corpusdir = "Topics/" + sourcedir
        corpus = doc2vec_model.doc2vec_train_corpus(corpusdir)
        #corpus = open(("./data/Topics/" + sourcedir + "/_doc2vec_corpus.cor"), "r")
        #print(corpus)
        model = doc2vec_model.train_doc2vec_model(corpus)
    else:
        return

    filenumber = 1  # File counter for use when iterating files and registering file ID

    for file in glob.iglob(os.path.join("./data/Topics sample/", sampledir, "*.json")):
        results = []
        with open(file) as f:
            json_article = pd.read_json(f, typ="series")
            content = json_article["content"]
            # print(content)
            if langmodel == "bow":
                sample_vec = vectorize.bow_vectorize(sourcedir, content)
                sample_dict = dict(sample_vec)
                for test_vec in corpus:
                    # print(test_vec)
                    test_dict = dict(test_vec)
                    # print(test_dict)
                    cos_sim = cosine_similarity(sample_dict, test_dict)
                    # print("\n", cos_sim)
                    results.append(int(cos_sim * 100))
            elif langmodel == "tfidf":
                sample_vec = vectorize.tfidf_vectorize(sourcedir, content)
                sample_dict = dict(sample_vec)
                for test_vec in corpus:
                    # print(test_vec)
                    test_dict = dict(test_vec)
                    # print(test_dict)
                    cos_sim = cosine_similarity(sample_dict, test_dict)
                    # print("\n", cos_sim)
                    results.append(int(cos_sim * 100))
            elif langmodel == "doc2vec":
                # This attempt to push doc2vec results thru the same comparison
                # pipeline as bow and tfidf isn't really working...
                content = content.split()
                sample_vec = doc2vec_model.infer_vector(model, content)
                # for test_vec in corpus:
                sims = model.docvecs.most_similar([sample_vec], topn=len(model.docvecs))
                results.append(sims[0])
            else:
                return


            # print("\n\n")
            results_series = pd.Series(results)
            comparison_df[str(filenumber)] = results_series.values
            filenumber += 1
            # Limit condition for use with larger file set
            # if filenumber == 1100:
            #    break

    if langmodel == "bow" or langmodel == "tfidf":

        # Display results as printed lists and histograms
        print("\n" + langmodel + " comparison\n")
        top_scores = []
        for column in comparison_df:
            print("Sample file " + str(column) + " is " + str(comparison_df[column].max())
                  + "% similar to corpus test file " + str((comparison_df[column].idxmax(axis=0) + 1)))
            top_scores.append([(comparison_df[column].max()),(comparison_df[column].idxmax(axis=0) + 1)])


        print("\n")
        comparison_df.plot.hist(alpha=0.5, bins=100, figsize=(10, 6))
        plt.xlim(0, 110)
        plt.ylim(0, 400)
        plt.show()

        #print("\n")
        #print(comparison_df)

        print("Finished", langmodel, "comparison, took {}".format(time.time() - start_time))

        return top_scores

    else:
        # break out columns of comparison df into two columns and print out results table
        # TO DO

        return comparison_df


def cosine_similarity(vector1, vector2):
    """This function reads in two vectors (dtype = dict)
    and returns their cosine similarity as a decimal fraction"""

    terms = set(vector1).union(vector2)
    dotprod = sum(vector1.get(k, 0) * vector2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(vector1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(vector2.get(k, 0)**2 for k in terms))
    if (magA == 0) or (magB == 0):
        return 0
    else:
        return dotprod / (magA * magB)


def compare_all_topics(sampledir, langmodel):
    """Opens the json files in the sample directory one by one, sends the article content
    for vectorizing, then performs cosine similarity against the corpuses in the all the source directories"""

    final_scores_df = pd.DataFrame()
    #final_scores_df.rename(columns={0: "% sim", 1: "ai"})

    directory = os.fsencode("./data/Topics/")

    for subdirectory in os.listdir(directory):
        subdirectoryname = os.fsdecode(subdirectory)

        if not subdirectoryname == ".DS_Store":
            print(str.upper(subdirectoryname))
            # plot_comparison.plot_comparison(subdirectoryname, subdirectoryname, "bow")
            top_scores = plot_comparison(subdirectoryname, sampledir, langmodel)
            top_scores_df = pd.DataFrame(top_scores)
            top_scores_df.loc[10] = [(top_scores_df[0].max()), 0]
            top_scores_df.columns = ["% sim", subdirectoryname]
            final_scores_df = pd.concat([final_scores_df, top_scores_df], axis=1)
            final_scores_df.rename(index={10: 'Best fit'})
            # print(final_scores_df)
            print("\n\n")

    return final_scores_df



if __name__ == "__main__":

    # Parser setup to create source_dir path
    parser = argparse.ArgumentParser()
    parser.prog = "DIRECTORIES_AND_MODEL"
    parser.description = "Get the source directory for the corpus, the sample directory for the " \
                         "test files, and the model desired (bow or tfidf)."
    parser.add_argument("get_sourcedir", help="Source directory")
    parser.add_argument("get_sampledir", help="Sample directory")
    parser.add_argument("get_langmodel", help="Desired language model ('bow' or 'tfidf')")
    args = parser.parse_args()
    sourcedir = str(args.get_sourcedir)
    sampledir = str(args.get_sampledir)
    langmodel = str(args.get_langmodel)
    print("Source directory for corpus is ./data/Topics/" + sourcedir)
    print("Sample jsons are taken from ./data/Topics sample/" + sampledir)
    print("Chosen language model is", langmodel)

    # Call the function to create the  dictionary and corpus from the data in the source directory
    plot_comparison(sourcedir, sampledir, langmodel)
    


