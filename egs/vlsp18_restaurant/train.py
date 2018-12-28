import pickle
from os.path import dirname, join
from time import time

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from load_data import load_dataset
from xgboost import XGBClassifier


def save_model(filename, clf):
    pickle.dump(clf, open(filename, 'wb'))


train_path = join(dirname(__file__), "data", "train.xlsx")
dev_path = join(dirname(__file__), "data", "dev.xlsx")
serialization_dir = join(dirname(__file__), "snapshots")
print("Load data...")
X_train, y_train = load_dataset(train_path)
X_dev, y_dev = load_dataset(dev_path)
target_names = list(set([i[0] for i in y_train]))

print("%d documents" % len(X_train))
print("%d categories" % len(target_names))

print("\nTraining model...")
t0 = time()
transformer = CountVectorizer(ngram_range=(1, 2), max_features=4000)
X_train = transformer.fit_transform(X_train)

y_transformer = MultiLabelBinarizer()
y_train = y_transformer.fit_transform(y_train)

model = OneVsRestClassifier(XGBClassifier(n_iter=500, max_depth=500))
estimator = model.fit(X_train, y_train)
t1 = time() - t0
print("Train time: %0.3fs" % t1)

print("\nEvaluate...")
y_dev = y_transformer.transform(y_dev)
X_dev = transformer.transform(X_dev)
y_pred = estimator.predict(X_dev)
print('F1 Score:', np.round(metrics.f1_score(y_dev, y_pred, average='micro'), 3))

print("\nSave model...")
t0 = time()
save_model(serialization_dir + "/x_transformer.pkl", transformer)
save_model(serialization_dir + "/y_transformer.pkl", y_transformer)
save_model(serialization_dir + "/model.pkl", estimator)
t1 = time() - t0
print("Save model time: %0.3fs" % t1)
