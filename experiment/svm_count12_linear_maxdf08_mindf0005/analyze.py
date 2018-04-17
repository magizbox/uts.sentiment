from os.path import join, dirname
import sys

from languageflow.board import Board
from languageflow.log import MultilabelLogger
from languageflow.log.count import CountLogger
from load_data import load_dataset
from model import sentiment

current_directory = sys.path[0]
data_file = join(dirname(dirname(dirname(current_directory))), "data", "fb_bank_sentiments_2200", "corpus", "test.xlsx")

X_test, y_test = load_dataset(data_file)
y_test = [tuple(item) for item in y_test]
y_pred = sentiment(X_test)

log_folder = join(dirname(__file__), "analyze")
model_folder = join(dirname(__file__), "model")

board = Board(log_folder=log_folder)

MultilabelLogger.log(X_test, y_test, y_pred, log_folder=log_folder)
CountLogger.log(model_folder=model_folder, log_folder=log_folder)

board.serve(port=62007)
