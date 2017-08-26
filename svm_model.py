import numpy as np
import time
import sklearn.datasets

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, grid_search
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics


def define_dataset(sourcedir):
    """Import the content data and define as a scikit learn dataset
    then Split the data for cross-validation purposes"""

    databunch = sklearn.datasets.load_files((open("./data/" + sourcedir))
                                            description = None,
                                            categories = None,
                                            load_content = True,
                                            shuffle = True,
                                            encoding = None,
                                            decode_error = 'strict',
                                            random_state = 0)

    print("The data categories are: ", databunch.target_names)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(databunch.data,
                                                        databunch.target,
                                                        test_size=.4)

    return databunch, X_train, X_test, y_train, y_test


def tfidf_extraction(xtrain, xtest):
    """Extract features from the dataset using tf-idf sparse vectorizer"""

    print("Extracting features from the training data using a sparse vectorizer")
    start_time = time.time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train_Tfidf = vectorizer.fit_transform(xtrain)
    print("Done in {}".format(time.time() - start_time))

    print("Extracting features from the test data using the same vectorizer")
    start_time = time.time()
    X_test_Tfidf = vectorizer.transform(xtest)
    print("Done in {}".format(time.time() - start_time))

    return X_train_Tfidf, X_test_Tfidf


def fit_nb(xtrain, ytrain):
    """Fit the data using a Naive Bayes model"""

    clf = MultinomialNB().fit(xtrain, ytrain)

    return clf


def fit_svm_linear(xtrain, ytrain):
    """Fit the data using a SVM linear kernel model"""

    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False).fit(xtrain, ytrain)

    return clf


def fit_svm_poly(xtrain, ytrain):
    """Fit the data using a SVM polynomial kernel model"""

    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma='auto', kernel='poly',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False).fit(xtrain, ytrain)

    return clf


def fit_svm_rbf(xtrain, ytrain):
    """Fit the data using a SVM radial basis function model"""

    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False).fit(xtrain, ytrain)

    return clf


def fit_svm_rbf_plus(xtrain, ytrain):
    """Fit the data using a SVM radial basis function model
    with higher degrees"""

    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=5, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False).fit(xtrain, ytrain)

    return clf



def svc_param_selection(X, y, nfolds):
    """Conduct a grid search of the available options to find the
    best parameters. Note that this took 24 hour on my aging
    2009 dual core MacBook!"""

    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['linear', 'poly', 'rbf']
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print(grid_search.best_params_)

    return


def fit_svm_best_params(xtrain, ytrain):
    """Fit the data using the parameters we had returned by our
    grid search"""

    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=5, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False).fit(xtrain, ytrain)

    return clf


def pipeline_tfidf_sgd(databunch, xtrain, xtest, ytrain, ytest):
    """Sklearn pipeline classifier, using tfidf vectors and a
    stochastic gradient descent method"""

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)), ])
    text_clf = text_clf.fit(xtrain, ytrain)
    predicted = text_clf.predict(xtest)
    print("\nMean prediction accuracy is: ", np.mean(predicted == ytest))
    print("\n\nClassification report:\n", metrics.classification_report(ytest, predicted, target_names = databunch.target_names))
    print("\n\nConfusion matrix:\n", metrics.confusion_matrix(ytest, predicted))

    return

from sklearn import svm, grid_search


def svc_param_selection(xtrain, ytrain, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['linear', 'poly', 'rbf']
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(xtrain, ytrain)
    grid_search.best_params_

    return grid_search.best_params_


def pipeline_tfidf_best_params(databunch, xtrain, xtest, ytrain, ytest):
    """Sklearn pipeline classifier, using tfidf vectors and a
    support vector machine tuned with our grid searched parameters"""

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
                                     decision_function_shape=None, degree=5, gamma='0.01', kernel='rbf',
                                     max_iter=-1, probability=False, random_state=None, shrinking=True,
                                     tol=0.001, verbose=False).fit(xtrain, ytrain)), ])
    text_clf = text_clf.fit(xtrain, ytrain)
    predicted = text_clf.predict(xtest)
    print("\nMean prediction accuracy is: ", np.mean(predicted == ytest))
    print("\n\nClassification report:\n", metrics.classification_report(ytest, predicted, target_names = databunch.target_names))
    print("\n\nConfusion matrix:\n", metrics.confusion_matrix(ytest, predicted))

    return


def predict_outcome_tfidf(clf, databunch, docs_new):
    """Predict the classification outcome on new documents by calling
    transform instead of fit_transform on the transformers, as they have by
    now been fitted to our main dataset"""

    X_new_tfidf = vectorizer.transform(docs_new)
    predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(docs_new, predicted):
        print('%r => %s \n' % (doc, databunch.target_names[category]))

    return

