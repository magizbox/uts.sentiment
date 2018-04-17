from os.path import join, dirname
from languageflow.board import Board
from languageflow.log import MultilabelLogger

from exported.linearsvc import sentiment
from load_data import load_dataset

data = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "hotel", "dev.xlsx")
X_dev, y_dev = load_dataset(data)
y_dev = [tuple(item) for item in y_dev]
y_pred = sentiment(X_dev)

log_folder = join(dirname(__file__), "analyze")

board = Board(log_folder=log_folder)

MultilabelLogger.log(X_dev, y_dev, y_pred, log_folder=log_folder)
# board.serve(port=62010)
