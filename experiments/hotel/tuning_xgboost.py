from os.path import dirname, join
from sklearn.feature_extraction.text import CountVectorizer
from load_data import load_dataset
from model import XGboostModel

data_train = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "hotel", "train.xlsx")
data_dev = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "hotel", "dev.xlsx")

X_train, y_train = load_dataset(data_train)
X_dev, y_dev = load_dataset(data_dev)
models = []

X = X_train + X_dev
y = y_train + y_dev

for n_iter in [100, 140, 160, 200, 500]:
    # for max_depth in [50, 100, 140, 160, 200, 300]:
    for max_depth in [200, 300, 400, 500]:
        # for max_features in [1000, 2000, 2200, 2400, 2600, 3000]:
        for max_features in [2000, 3000, 4000]:
            name = "XGBoost(n_iter {0} max_depth {1}) + Count(bigram, max_features {2})".format(n_iter, max_depth, max_features)
            params = {"n_iter": n_iter, "max_depth": max_depth}
            model = XGboostModel(
                name,
                params,
                CountVectorizer(ngram_range=(1, 2), max_features=max_features)
            )
            models.append(model)

for model in models:
    from datetime import datetime
    start = datetime.now()
    model.load_data(X, y)
    model.fit_transform()
    model.train()
    model.evaluate(X_dev, y_dev)
    # model.export()
    print(datetime.now() - start)
