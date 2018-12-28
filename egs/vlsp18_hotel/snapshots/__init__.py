import pickle
import sys
from os.path import join, abspath, dirname

from normalize import normalize_text

cwd = dirname(abspath(__file__))
sys.path.append(dirname(dirname(cwd)))

x_transformer_file = open(join(cwd, "x_transformer.pkl"), "rb")
x_transformer = pickle.load(x_transformer_file)
y_transformer_file = open(join(cwd, "y_transformer.pkl"), "rb")
y_transformer = pickle.load(y_transformer_file)
estimator_file = open(join(cwd, "model.pkl"), "rb")
estimator = pickle.load(estimator_file)


def sentiment(text):
    text = normalize_text(text)
    X = x_transformer.transform([text])
    y_pred = estimator.predict(X)
    labels = y_transformer.inverse_transform(y_pred)[0]
    return labels
