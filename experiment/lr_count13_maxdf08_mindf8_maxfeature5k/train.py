from os.path import dirname, join
import sys
from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.transformer.count import CountVectorizer
from sklearn.linear_model import LogisticRegression
from languageflow.validation.validation import TrainTestSplitValidation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import load_dataset

if __name__ == '__main__':
    current_directory = sys.path[0]
    data_file = join(dirname(dirname(current_directory)), "data", "fb_bank_sentiments_2200", "corpus", "train.xlsx")
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.data(X, y)

    transformer = CountVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=8, max_features=5000)
    flow.transform(MultiLabelBinarizer())
    flow.transform(transformer)
    flow.add_model(Model(OneVsRestClassifier(LogisticRegression()), "LogisticRegression"))
    flow.set_validation(TrainTestSplitValidation(test_size=0.1))

    flow.train()
    flow.export(model_name="LogisticRegression", export_folder="model")
