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

ex = Experiment('with_emoticons_full')
ex.observers.append(MongoObserver.create())

negative_emoticons = {':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{'}

positive_emoticons = {'=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '✌', '✨', '❣', '❤', '🌝', '🌷', '🌸',
                      '🌺', '🌼', '🍓', '🎈', '🐅', '🐶', '🐾', '👉', '👌', '👍', '👏', '👻', '💃', '💄', '💋',
                      '💌', '💎', '💐', '💓', '💕', '💖', '💗', '💙', '💚', '💛', '💜', '💞', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)'}


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


class CountEmoticons(BaseEstimator, TransformerMixin):
    def count_emoticon(self, s):
        positive_count = 0
        negative_count = 0
        for emoticon in positive_emoticons:
            positive_count += s.count(emoticon)
        for emoticon in negative_emoticons:
            negative_count += s.count(emoticon)
        return positive_count, negative_count

    def transform(self, x):
        return [self.count_emoticon(s) for s in x]

    def fit(self, x, y=None):
        return self


@ex.main
def my_run(estimator__C,
           features__lower_pipe__tfidf__ngram_range,
           features__with_tone_char__ngram_range,
           features__remove_tone__tfidf__ngram_range):
    params = locals().copy()
    start = time.time()
    print(params)
    corpus: CategorizedCorpus = DataFetcher.load_corpus(NLPData.VLSP2016_SA)
    pipeline = Pipeline(
        steps=[
            ('features', FeatureUnion([
                ('lower_pipe', Pipeline([
                    ('lower', Lowercase()),
                    ('tfidf', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))])),
                ('with_tone_char', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char')),
                ('remove_tone', Pipeline([
                    ('remove_tone', RemoveTone()),
                    ('lower', Lowercase()),
                    ('tfidf', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))])),
                ('emoticons', CountEmoticons())
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
    print(time.time() - start)
    return score['dev_score']


best_score = 1.0


def objective(space):
    global best_score
    test_score = ex.run(config_updates=space).result
    score = 1 - test_score
    print("Score:", score)
    return score


space = {
    'estimator__C': hp.choice('C', np.arange(0.005, 1.0, 0.005)),
    'features__lower_pipe__tfidf__ngram_range': hp.choice('features__lower_pipe__lower_tfidf__ngram_range',
                                                          [(1, 2), (1, 3), (1, 4)]),
    'features__with_tone_char__ngram_range': hp.choice('features__with_tone_char__ngram_range',
                                                       [(1, 4), (1, 5), (1, 6)]),
    'features__remove_tone__tfidf__ngram_range': hp.choice('features__remove_tone__tfidf__ngram_range',
                                                           [(1, 2), (1, 3), (1, 4)])
}
start = time.time()
trials = Trials()
best = fmin(objective, space=space, algo=tpe.suggest, max_evals=300, trials=trials)

print("Hyperopt search took %.2f seconds for 200 candidates" % ((time.time() - start)))
print(-best_score, best)
