from tempfile import mkdtemp

import joblib
import unidecode
from languageflow.data import CategorizedCorpus
from languageflow.data_fetcher import DataFetcher, NLPData
from languageflow.models.text_classifier import TextClassifier, TEXT_CLASSIFIER_ESTIMATOR
from languageflow.trainers.model_trainer import ModelTrainer
from sacred import Experiment
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sacred.observers import MongoObserver

ex = Experiment('vlsp2016')

ex.observers.append(MongoObserver.create())

negative_emoticons = {':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{'}

positive_emoticons = {'=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '✌', '✨', '❣', '❤', '🌝', '🌷', '🌸',
                      '🌺', '🌼', '🍓', '🎈', '🐅', '🐶', '🐾', '👉', '👌', '👍', '👏', '👻', '💃', '💄', '💋',
                      '💌', '💎', '💐', '💓', '💕', '💖', '💗', '💙', '💚', '💛', '💜', '💞', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)'}
@ex.config
def config():
    estimator_C = 0.375
    lower_tfidf__ngram_range = (1, 3)
    with_tone_char__ngram_range = (1, 5)
    remove_tone__tfidf__ngram_range = (1, 2)


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


@ex.automain
def run(estimator_C, lower_tfidf__ngram_range,  with_tone_char__ngram_range, remove_tone__tfidf__ngram_range):
    corpus: CategorizedCorpus = DataFetcher.load_corpus(NLPData.VLSP2016_SA)
    pipeline = Pipeline(
        steps=[
            ('features', FeatureUnion([
                ('lower_tfidf', Pipeline([
                    ('lower', Lowercase()),
                    ('tfidf', TfidfVectorizer(ngram_range=lower_tfidf__ngram_range, norm='l2', min_df=2))])),
                ('with_tone_char', TfidfVectorizer(ngram_range=with_tone_char__ngram_range, norm='l2', min_df=2, analyzer='char')),
                ('remove_tone', Pipeline([
                    ('remove_tone', RemoveTone()),
                    ('lower', Lowercase()),
                    ('tfidf', TfidfVectorizer(ngram_range=remove_tone__tfidf__ngram_range, norm='l2', min_df=2))])),
                ('emoticons', CountEmoticons())
            ])),
            ('estimator', SVC(kernel='linear', C=estimator_C, class_weight=None, verbose=True))
        ]
    )
    classifier = TextClassifier(estimator=TEXT_CLASSIFIER_ESTIMATOR.PIPELINE, pipeline=pipeline)
    model_trainer = ModelTrainer(classifier, corpus)
    tmp_model_folder = mkdtemp()

    def negative_f1_score(y_true, y_pred):
        score_class_0, score_class_1, score_class_2 = f1_score(y_true, y_pred, average=None)
        return score_class_1

    def macro_f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    score = model_trainer.train(tmp_model_folder, scoring=negative_f1_score)
    ex.log_scalar('dev_score', score['dev_score'])
    ex.log_scalar('test_score', score['test_score'])
    return score['test_score']