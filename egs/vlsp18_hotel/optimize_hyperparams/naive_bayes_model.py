import warnings
from os.path import join, dirname

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from load_data import load_dataset
from model import ClassifierModel

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    train_path = join(dirname(dirname(__file__)), "data", "train.xlsx")
    dev_path = join(dirname(dirname(__file__)), "data", "dev.xlsx")
    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(dev_path)

    models = [
        ClassifierModel("Tfidf Bigram", TfidfVectorizer(ngram_range=(1, 2))),
        ClassifierModel("Tfidf Trigram", TfidfVectorizer(ngram_range=(1, 3))),
        ClassifierModel("Count Bigram", CountVectorizer(ngram_range=(1, 2))),
        ClassifierModel("Count Trigram", CountVectorizer(ngram_range=(1, 3)))
    ]

    for n in [2000, 5000, 10000, 15000, 20000]:
        model = ClassifierModel(
            "Count Max Feature {}".format(n),
            CountVectorizer(max_features=n)
        )
        models.append(model)

    for n in [2000, 5000, 10000, 15000, 20000]:
        model = ClassifierModel(
            "Tfidf Max Feature {}".format(n),
            TfidfVectorizer(max_features=n)
        )
        models.append(model)

    for n in [500, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        for ngram in [('Bigram', (1, 2)), ("Trigram", (1, 3))]:
            model = ClassifierModel(
                "Count {0} + Max Feature {1}".format(ngram[0], n),
                CountVectorizer(ngram_range=ngram[1], max_features=n)
            )
            models.append(model)

    for n in [500, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        for ngram in [('Bigram', (1, 2)), ("Trigram", (1, 3))]:
            model = ClassifierModel(
                "Tfidf {0} + Max Feature {1}".format(ngram[0], n),
                TfidfVectorizer(ngram_range=ngram[1], max_features=n)
            )
            models.append(model)

    for model in models:
        model.load_data(X_train, y_train, X_test, y_test)
        model.fit_transform()
        model.train(clf_model=MultinomialNB())
        model.evaluate(model_name="MultinomialNB")
