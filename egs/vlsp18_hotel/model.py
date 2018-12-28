from os.path import join, dirname

import numpy as np
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier


class ClassifierModel:
    def __init__(self, vect, transformer):
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.name = vect
        self.y_transformer = MultiLabelBinarizer()
        self.transformer = transformer
        self.estimator = None

    def load_data(self, X_train, y_train, X_dev, y_dev):
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev

    def show_info(self):
        print("=======")
        print(self.name)

    def fit_transform(self):
        self.X_train = self.transformer.fit_transform(self.X_train)
        self.y_train = self.y_transformer.fit_transform(self.y_train)

    def train(self, clf_model):
        self.show_info()
        model = OneVsRestClassifier(clf_model)
        self.estimator = model.fit(self.X_train, self.y_train)

    def _create_log_file_name(self, score, model_name):
        file_name = "logs/" + \
                    "{:.4f}".format(score) + \
                    "_" + \
                    model_name + \
                    "_" + \
                    self.name + \
                    ".txt"
        return file_name

    def evaluate(self, model_name):
        y_dev = self.y_transformer.transform(self.y_dev)
        X_dev = self.transformer.transform(self.X_dev)
        y_predict = self.estimator.predict(X_dev)
        score = np.round(metrics.f1_score(y_dev, y_predict, average='micro'), 5)
        print("F1 micro: ", score)
        log_file = self._create_log_file_name(score, model_name)
        log_file = join(dirname(__file__), log_file)
        with open(log_file, "w") as f:
            f.write("")


class XGboostModel:
    def __init__(self, name, params, transformer):
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.name = name
        self.y_transformer = MultiLabelBinarizer()
        self.transformer = transformer
        self.params = params
        self.model = OneVsRestClassifier(XGBClassifier(**params))

    def load_data(self, X_train, y_train, X_dev, y_dev):
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev

    def show_info(self):
        print("=======")
        print(self.name)

    def fit_transform(self):
        self.X_train = self.transformer.fit_transform(self.X_train)
        self.y_train = self.y_transformer.fit_transform(self.y_train)

    def train(self):
        self.show_info()
        self.model.fit(self.X_train, self.y_train)

    def _create_log_file_name(self, score, model_name):
        file_name = "logs/" + \
                    "{:.4f}".format(score) + \
                    "_" + \
                    model_name + \
                    ".txt"
        return file_name

    def evaluate(self, model_name):
        y_dev = self.y_transformer.transform(self.y_dev)
        X_dev = self.transformer.transform(self.X_dev)
        y_predict = self.model.predict(X_dev)
        score = np.round(metrics.f1_score(y_dev, y_predict, average='micro'), 5)
        print("f1 micro: ", score)
        log_file = self._create_log_file_name(score, model_name)
        log_file = join(dirname(__file__), log_file)
        with open(log_file, "w") as f:
            f.write("")
