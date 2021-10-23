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
               |      \                       /                      \
           Standing   Sitting          Constant  Change                    Short Lived transition
              (8)         (7)            /   |            \                   /             |          \             
                                  Walking   Running  Continuous           Fall           Bending    Getting Up Fast
                                     (9)      (5)         Jumping        /  |  \          (1)            (3)
                                                          (4)           /   |   \
                                                                  Forward  Side  Backward
                                                                  Fall     Fall  Fall
                                                                  (2)      (6)       (0)                   
                                
    """
    class_hierarchy = {
        ROOT: ["Static Postures", "Dynamic Transitions"],
        "Static Postures": ["8","7"],
        "Dynamic Transitions": ["Constant Change", "Short Lived Transitions"],
        "Constant Change": ["9", "5","4"],
        "Short Lived Transitions": ["Fall","1","3"],
        "Fall":["2","6","0"]
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
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n",report )
    label_to_names={0: 'Back Fall', 1: 'Bending', 2: 'Front Fall', 3: 'Getting Up Fast', 4: 'Jumping',
                    5: 'Running', 6: 'Side Fall', 7: 'Sitting', 8: 'Standing', 9:'Walking' }
    print(label_to_names)
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


