from sklearn.datasets import make_classification
import pandas as pd

X,Y=make_classification(n_samples=200,
                        n_features=25,
                        n_informative=2,
                        n_redundant=10,
                        n_classes=2,
                        random_state=8)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, rand,STATUS_OK, Trials,space_eval
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2,f_classif

best_score = 1.0


def objective01(space):
    global best_score
    global score_01

    keys01 = ('penalty', 'C', 'solver')
    keys02 = ('n_estimators', 'bootstrap')

    # This is the most pythonic way I know of to split dictionaries
    subspace01 = {k: space[k] for k in set(space).intersection(keys01)}
    subspace02 = {k: space[k] for k in set(space).intersection(keys02)}

    model = BaggingClassifier(base_estimator=SVC(**subspace01),
                              max_samples=(1.0 / (subspace02.get('n_estimators'))),
                              **subspace02)

    kfold = KFold(n_splits=3, random_state=1985, shuffle=True)
    score_01 = -cross_val_score(model, X, Y, cv=kfold, scoring='neg_log_loss', verbose=False).mean()

    if (score_01 < best_score):
        best_score = score_01

    return score_01


best_score = 1.0


def objective02(space):
    global best_score
    global score_02

    keys = ('penalty', 'C', 'solver')

    subspace = {k: space[k] for k in set(space).intersection(keys)}

    model = SVC(probability=True, **subspace, )

    kfold = KFold(n_splits=3, random_state=1985, shuffle=True)
    score_02 = -cross_val_score(model, X, Y, cv=kfold, scoring='neg_log_loss', verbose=False).mean()

    if (score_02 < best_score):
        best_score = score_02

    return score_02

space={
    'C': hp.choice('x_C', np.arange(0.0,1.0,0.00005)),
    'kernel': hp.choice('x_kernel',['linear', 'poly', 'rbf']),
    'degree': hp.choice('x_degree',[2,3,4]),
    'probability': hp.choice('x_probability',[True]),
    'n_estimators': hp.choice('x_n_estimators',np.arange(5,21,1)),
    'bootstrap': hp.choice('x_bootstrap',[False,True])
    }

# start=time.time()
# trials=Trials()
# max_evals=20
# best=fmin(objective01,
#           space=space,
#           algo=tpe.suggest,
#           max_evals=max_evals,
#           trials=trials)
#
# print("Hyperopt search took %.2f seconds for %.2f candidates" % ((time.time() - start),max_evals))
# print(-best_score, space_eval(space,best))

start=time.time()
trials=Trials()
max_evals=20
best=fmin(objective02,
          space=space,
          algo=tpe.suggest,
          max_evals=max_evals,
          trials=trials)

print("Hyperopt search took %.2f seconds for %.2f candidates" % ((time.time() - start),max_evals))
print(-best_score, space_eval(space,best))