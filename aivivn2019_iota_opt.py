from tempfile import mkdtemp

import joblib
import time
import unidecode
from hyperopt import Trials, fmin, hp, tpe
from languageflow.data import CategorizedCorpus
from languageflow.data_fetcher import DataFetcher, NLPData
from languageflow.models.text_classifier import TextClassifier, TEXT_CLASSIFIER_ESTIMATOR
from languageflow.trainers.model_trainer import ModelTrainer
from sacred import Experiment
from sacred.optional import np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sacred.observers import MongoObserver

ex = Experiment('aivivn')
ex.observers.append(MongoObserver.create())



class Lowercase(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [s.lower() for s in x]

    def fit(self, x, y=None):
        return self


class RemoveTone(BaseEstimator, TransformerMixin):
    def remove_tone(self, s):
        return unidecode.unidecode(s)

    def transform(self, x):
        return [self.remove_tone(s) for s in x]

    def fit(self, x, y=None):
        return self

import inspect
@ex.main
def my_run(estimator__C):
    params = locals().copy()
    print(params)
    corpus: CategorizedCorpus = DataFetcher.load_corpus(NLPData.AIVIVN2019_SA)
    pipeline = Pipeline(
        steps=[
            ('features', FeatureUnion([
                ('lower_tfidf', Pipeline([
                    ('lower', Lowercase()),
                    ('tfidf', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))])),
            #     ('with_tone_char', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char')),
            #     ('remove_tone', Pipeline([
            #         ('remove_tone', RemoveTone()),
            #         ('lower', Lowercase()),
            #         ('tfidf', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))]))
            ])),
            ('estimator', SVC(kernel='linear', C=0.2175, class_weight=None, verbose=True))
        ]
    )
    pipeline.set_params(**params)
    classifier = TextClassifier(estimator=TEXT_CLASSIFIER_ESTIMATOR.PIPELINE, pipeline=pipeline)
    model_trainer = ModelTrainer(classifier, corpus)
    tmp_model_folder = mkdtemp()

    def negative_f1_score(y_true, y_pred):
        score_class_0, score_class_1 = f1_score(y_true, y_pred, average=None)
        return score_class_1

    def macro_f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    score = model_trainer.train(tmp_model_folder, scoring=negative_f1_score)
    ex.log_scalar('dev_score', score['dev_score'])
    ex.log_scalar('test_score', score['test_score'])
    return score['test_score']


best_score = 1.0


def objective(space):
    global best_score
    test_score = ex.run(config_updates=space).result
    score = 1 - test_score
    print("Score:", score)
    return score


space = {
    'estimator__C': hp.choice('C', np.arange(0.005, 1.0, 0.1)),
}
start = time.time()
trials = Trials()
best = fmin(objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

print("Hyperopt search took %.2f seconds for 200 candidates" % ((time.time() - start)))
print(-best_score, best)