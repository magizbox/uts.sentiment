from os.path import join, dirname
import joblib
import sys

sys.path.insert(0, dirname(__file__))

x_transform = joblib.load(join(dirname(__file__), "tfidf.transformer.bin"))
y_transform = joblib.load(join(dirname(__file__), "y_transformer.bin"))
estimator = joblib.load(join(dirname(__file__), "model.bin"))


def sentiment(X):
    if isinstance(X, list):
        X = x_transform.transform(X)
        y = estimator.predict(X)
        y = y_transform.inverse_transform(y)
        return y
    else:
        X = x_transform.transform([X])
        y = estimator.predict(X)
        y = y_transform.inverse_transform(y)
        return y
