from os.path import join, dirname

from exported.linearsvc_full import sentiment
from load_data import load_dataset
from score import multilabel_f1_score

test_data = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "restaurant", "test.xlsx")
test_gold_data = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "restaurant", "test-gold-data.xlsx")
X_test, y_test = load_dataset(test_data)
X_test_gold, y_test_gold = load_dataset(test_gold_data)

y_test_gold = [tuple(item) for item in y_test_gold]
y_predict = sentiment(X_test)
f1 = multilabel_f1_score(y_test_gold, y_predict)
print(0)
