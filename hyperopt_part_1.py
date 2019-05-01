from sklearn.datasets import make_classification
import pandas as pd

X, Y = make_classification(n_samples=300,
                           n_features=25,
                           n_informative=2,
                           n_redundant=10,
                           n_classes=2,
                           random_state=8)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import hp, fmin, tpe
from sklearn.svm import SVC
import time
import numpy as np

classifier = SVC(gamma='auto')

params = {
    'C': np.arange(0.005, 1.0, 0.005),
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2, 3, 4],
    'probability': [True],
}

n_iter = 200
random_search = RandomizedSearchCV(classifier,
                                   param_distributions=params,
                                   n_iter=n_iter,
                                   scoring='neg_log_loss')
start = time.time()
random_search.fit(X, Y)

print("RandomizedSearchCV took %.2f seconds for %d candidates" % ((time.time() - start), n_iter))
print(random_search.best_score_, random_search.best_params_)

grid_search = GridSearchCV(classifier,
                           param_grid=params,
                           scoring='neg_log_loss')

start = time.time()
grid_search.fit(X, Y)

print("GridSearchCV took %.2f seconds" % ((time.time() - start)))
print(grid_search.best_score_, grid_search.best_params_)

best_score = 1.0


def objective(space):
    global best_score
    model = SVC(**space)
    kfold = KFold(n_splits=3, random_state=1985, shuffle=True)
    score = -cross_val_score(model, X, Y, cv=kfold, scoring='neg_log_loss', verbose=False).mean()

    if (score < best_score):
        best_score = score

    return score


space = {
    'C': hp.choice('C', np.arange(0.005, 1.0, 0.005)),
    'kernel': hp.choice('x_kernel', ['linear', 'poly', 'rbf']),
    'degree': hp.choice('x_degree', [2, 3, 4]),
    'probability': hp.choice('x_probability', [True])
}

start = time.time()
trials = Trials()
best = fmin(objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

print("Hyperopt search took %.2f seconds for 200 candidates" % ((time.time() - start)))
print(-best_score, best)
