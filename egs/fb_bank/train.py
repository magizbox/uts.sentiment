import pickle
from os.path import dirname, join
from time import time

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
import numpy as np

from load_data import load_dataset


def save_model(filename, clf):
    pickle.dump(clf, open(filename, 'wb'))


train_path = join(dirname(__file__), "data", "train.xlsx")
test_path = join(dirname(__file__), "data", "test.xlsx")
serialization_dir = join(dirname(__file__), "snapshots")
print("Load data...")
X_train, y_train = load_dataset(train_path)
X_test, y_test = load_dataset(test_path)
target_names = list(set([i[0] for i in y_train]))

print("%d documents" % len(X_train))
print("%d categories" % len(target_names))

print("\nTraining model...")
t0 = time()
transformer = CountVectorizer(ngram_range=(1, 3))
X_train = transformer.fit_transform(X_train)

y_transformer = MultiLabelBinarizer()
y_train = y_transformer.fit_transform(y_train)

model = OneVsRestClassifier(LinearSVC())
estimator = model.fit(X_train, y_train)
t1 = time() - t0
print("Train time: %0.3fs" % t1)

print("\nEvaluate...")
y_test = y_transformer.transform(y_test)
X_test = transformer.transform(X_test)
y_pred = estimator.predict(X_test)
print('F1 Score:', np.round(metrics.f1_score(y_test, y_pred, average='micro'), 3))

print("\nSave model...")
t0 = time()
save_model(serialization_dir + "/x_transformer.pkl", transformer)
save_model(serialization_dir + "/y_transformer.pkl", y_transformer)
save_model(serialization_dir + "/model.pkl", estimator)
t1 = time() - t0
print("Save model time: %0.3fs" % t1)
