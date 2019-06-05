import os
import shutil

from languageflow.data import CategorizedCorpus
from languageflow.data_fetcher import DataFetcher, NLPData
from languageflow.models.text_classifier import TextClassifier, TEXT_CLASSIFIER_ESTIMATOR
from languageflow.trainers.model_trainer import ModelTrainer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from time import time

from text_features import Lowercase, RemoveTone, CountEmoticons, Tokenrize, RemoveDuplicate

start = time()

model_folder = "tmp/sentiment_svm_aivivn2019"
try:
    shutil.rmtree(model_folder)
except:
    pass
finally:
    os.makedirs(model_folder)

estimator_C = 0.16
lower_tfidf__ngram_range = (1, 2)
with_tone_char__ngram_range = (1, 5)
remove_tone__tfidf__ngram_range = (1, 2)

print(">>> Train AIVIVN2019 Sentiment Analysis")
corpus: CategorizedCorpus = DataFetcher.load_corpus(NLPData.AIVIVN2019_SA)
print("\n\n>>> Sample sentences")
for s in corpus.train[:10]:
    print(s)

pipeline = Pipeline(
    steps=[
        ('features', FeatureUnion([
            ('lower_tfidf', Pipeline([
                ('token', Tokenrize()),
                ('lower', Lowercase()),
                ('remove_duplicate', RemoveDuplicate()),
                ('tfidf', TfidfVectorizer(ngram_range=lower_tfidf__ngram_range, norm='l2', min_df=2))])),
            ('with_tone_char',
             TfidfVectorizer(ngram_range=with_tone_char__ngram_range, norm='l2', min_df=2, analyzer='char')),
            ('remove_tone', Pipeline([
                ('remove_tone', RemoveTone()),
                ('lower', Lowercase()),
                ('tfidf', TfidfVectorizer(ngram_range=remove_tone__tfidf__ngram_range, norm='l2', min_df=2))])),
            ('emoticons', CountEmoticons())
        ])),
        ('estimator', SVC(kernel='linear', C=estimator_C, class_weight=None, verbose=True))
    ]
)

print("\n\n>>> Start training")
classifier = TextClassifier(estimator=TEXT_CLASSIFIER_ESTIMATOR.PIPELINE, pipeline=pipeline)
model_trainer = ModelTrainer(classifier, corpus)


def macro_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


model_trainer.train(model_folder, scoring=macro_f1_score)
print("\n\n>>> Finish training")
print(f"Your model is saved in {model_folder}")
print(f"Running in {time() - start}")