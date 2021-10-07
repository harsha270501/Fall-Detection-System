from sklearn import svm
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled
#from sklearn_hierarchical_classification.tests.fixtures import make_digits_dataset

def make_digits_dataset(targets=None, as_str=True):
    X, y = load_digits(return_X_y=True)
    if targets:
        ix = np.isin(y, targets)
        X, y = X[np.where(ix)], y[np.where(ix)]

    if as_str:
        # Convert targets (classes) to strings
        y = y.astype(str)

    return X, y

def hierarchical_model_classify(X_train,X_test,y_train,y_test):
    r"""Test that a nontrivial hierarchy leaf classification behaves as expected.
    We build the following class hierarchy along with data from the handwritten digits dataset:
                            <ROOT>
                /                              \
          Static                                Dynamic 
             postures                               transitions  
        /     |      \                       /                      \
Fall-like  Standing   Sitting          Constant  Change                    Short Lived transition
activity     (7)         (6)            /   |            \                   /             |          \             
(lying down)                     Walking   Running  Continuous           Fall           Bending    Getting Up Fast
    (0)                              (8)      (5)         Jumping        /    \          (3)            (9)
                                                          (4)           /      \
                                                                  Forward       Backward
                                                                  Fall          Fall
                                                                  (1)            (2)                   
                                
    """
    class_hierarchy = {
        ROOT: ["Static Postures", "Dynamic Transitions"],
        "Static Postures": ["7","6"],
        "Dynamic Transitions": ["Constant Change", "Short Lived Transitions"],
        "Constant Change": ["8", "5","4"],
        "Short Lived Transitions": ["Fall","3","9"],
        "Fall":["1","2"]
    }
    """
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": ["1", "7"],
        "B": ["C", "9"],
        "C": ["3", "8"],
    }"""
    base_estimator = make_pipeline(
        TruncatedSVD(n_components=24),
        svm.SVC(
            gamma=0.001,
            kernel="rbf",
            probability=True
        ),
    )
    clf = HierarchicalClassifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
    )
    """
    X, y = make_digits_dataset(
        targets=[1, 7, 3, 8, 9],
        as_str=False,
    )
    # cast the targets to strings so we have consistent typing of labels across hierarchy
    y = y.astype(str)
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )"""
    y_train=y_train.astype(str)
    y_test=y_test.astype(str)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Demonstrate using our hierarchical metrics module with MLB wrapper
    with multi_labeled(y_test, y_pred, clf.graph_) as (y_test_, y_pred_, graph_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
        )
        print("h_fbeta_score: ", h_fbeta)
    
    accuracy= accuracy_score(y_test,y_pred)
    return accuracy


