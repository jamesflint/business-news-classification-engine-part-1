import gensim
import sklearn.datasets

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


docs_new = ["UK government may be planning diesel-scrappage scheme<p>The UK's Department for Transport and the Department for Environment, Food and Rural Affairs is in talks to introduce a diesel vehicle scrappage scheme as the country searches for ways to cut carbon emissions and pollution, according to unnamed sources. The strategy would offer a discount on low-emissions cars or cash back to drivers who choose to trade in their old diesel vehicles. A Department for Transport spokesperson said there are presently no plans to introduce a scrappage scheme. However, Transport Secretary Chris Grayling has shown support for the measures in industry meetings, according to the Telegraph.</p>"
, "UK-led consortium looks to robots to monitor windfarms and cables<p>A UK-led consortium is launching a $5m project to use robots for monitoring offshore wind farms and undersea cabling. The Engineering and Physical Sciences Research Council (EPSRC)-backed scheme hopes to improve how wind farms are maintained and managed. The project is developing \"novel sensing techniques\" to inspect systems in the field, said Dr David Flynn, director of Smart Systems Group at Heriot-Watt University, a member of the consortium. These include a dolphin-inspired local frequency sonar attached to an autonomous vehicle that can be used to inspect undersea cables, and drones to inspect offshore substations.</p>\n",
            "Ubtech partners with Amazon's Alexa in new robot<p>Ubtech Robotics has partnered with Amazon to bring Alexa, Amazon’s virtual assistant, to its latest robot, Lynx. Owners of Lynx will be able to interact with it as though it were a personal assistant. Since its launch, Amazon has sold millions of Echo devices, but lately they have begun to free Alexa from her Echo chamber, installing the natural language-processing assistant in home devices, wearable technology, cars and even robots. Amazon seems to be using partners such as Ubtech to redefine the physical form of its cloud-based technology.</p>\n"]

databunch = sklearn.datasets.load_files("./data/SVM topics/",
                                            description=None,
                                            categories=None,
                                            load_content=True,
                                            shuffle=True,
                                            encoding=None,
                                            decode_error='strict',
                                            random_state=0)


def read_corpus_svm(corpus, tokens_only=False):
    '''Pre-process the textual content for training the doc2vec model'''
    for i, item in enumerate(corpus):
            if tokens_only:
                yield gensim.utils.simple_preprocess(item)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(item), [i])
    return


if __name__ == "__main__":

    # Split the data for cross-validation purposes
    X_train, X_test, y_train, y_test = train_test_split(databunch.data, databunch.target, test_size=.4)

    #Turn the cross validation splits into vectors
    X_train_corpus = list(read_corpus_svm(X_train))
    X_test_corpus = list(read_corpus_svm(X_test, tokens_only=True))
    X_new_corpus = list(read_corpus_svm(docs_new, tokens_only=True))

    #Create & train the doc2vec model
    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=10)
    model.build_vocab(X_train_corpus)
    model.train(X_train_corpus, total_examples=model.corpus_count, epochs=model.iter)

    #Infer the doc2vec vectors for use with the SVM
    X_train_doc2vec = []
    for doc_id in range(len(X_train_corpus)):
        inferred_vector = model.infer_vector(X_train_corpus[doc_id].words)
        X_train_doc2vec.append(inferred_vector)

    X_test_doc2vec = []
    for doc_id in range(len(X_test_corpus)):
        inferred_vector = model.infer_vector(X_test_corpus[doc_id])
        X_test_doc2vec.append(inferred_vector)

    X_new_doc2vec = []
    for doc_id in range(len(X_new_corpus)):
        inferred_vector = model.infer_vector(X_new_corpus[doc_id])
        X_new_doc2vec.append(inferred_vector)

    # Run SVC with the grid-searched parameters
    clf = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=5, gamma=0.1, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False).fit(X_train_doc2vec, y_train)

    # Print out results
    predicted = clf.predict(X_new_doc2vec)
    for doc, category in zip(docs_new, predicted):
        print('%r => %s \n' % (doc, databunch.target_names[category]))
    predicted = clf.predict(X_test_doc2vec)
    print("\nMean prediction accuracy is: ", np.mean(predicted == y_test))
    print("\n\nClassification report:\n", metrics.classification_report(y_test,
                                                                        predicted,
                                                                        target_names=databunch.target_names))
    print("\n\nConfusion matrix:\n", metrics.confusion_matrix(y_test, predicted))

