from os.path import dirname, join
from sklearn.feature_extraction.text import CountVectorizer
from load_data import load_dataset
from model import XGboostModel

data_train = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "hotel", "train.xlsx")
data_dev = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "hotel", "dev.xlsx")

X_train, y_train = load_dataset(data_train)
X_dev, y_dev = load_dataset(data_dev)

params = {"n_iter": 140, "max_depth": 400}
model = XGboostModel("Count Trigram", params, CountVectorizer(ngram_range=(1, 2), max_features=2000))
model.load_data(X_train, y_train)
model.fit_transform()
model.train()
model.evaluate(X_dev, y_dev)
model.export(folder="exported/xgboost")
