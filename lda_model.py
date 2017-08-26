import time

from gensim import corpora
from gensim.models import LdaMulticore

DICTIONARY = corpora.Dictionary.load('./data/All topics/_bow_corpus.dict')
CORPUS = corpora.MmCorpus('./data/All topics/_bow_corpus.mm')


def lda_model_onepass(dictionary, corpus, topics):
    """Create a single pass LDA model"""

    start_time = time.time()
    model = LdaMulticore(corpus, id2word = dictionary, num_topics = topics)
    model.save(""./data/lda/all_topics_single.lda"")
    print(model.print_topics(-1))
    print("\nDone in {}".format(time.time() - start_time))

    return model


def lda_model_multipass(dictionary, corpus, topics, num_passes):
    """Create a multiple pass LDA model"""

    start_time = time.time()
    model = LdaMulticore(corpus, id2word = dictionary, num_topics = topics, passes = num_passes)
    model.save("./data/lda/all_topicsx" + str(num_passes) + ".lda")
    print(model.print_topics(-1))
    print("\nDone in {}".format(time.time() - start_time))

    return model


def comparison(dictionary, query):
    """Transform the query into a bow vector"""

    query = query.split()
    query = dictionary.doc2bow(query)

    return query


def run_query(model, dictionary, doc_string):
    """Handle a query procedure matching a given text to the LDA topics"""

    vector = comparison(dictionary, doc_string)
    # model[vector]
    a = list(sorted(model[vector], key=lambda x: x[1]))
    print(a[0])
    print(model.print_topic(a[0][0]))  # least related
    print(a[-1])
    print(model.print_topic(a[-1][0]))  # most related

    return

