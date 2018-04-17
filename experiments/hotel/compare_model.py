from os.path import join, dirname
from languageflow.board import Board
from languageflow.log import MultilabelLogger
import json


def load_data(file):
    with open(file) as f:
        content = json.load(open(file))
        X, y = content["text"], content["labels"]
        y = [tuple(item) for item in y]
        return X, y


X, y1 = load_data("results/svc_full.json")
X, y2 = load_data("results/svc.json")

log_folder = join(dirname(__file__), "compare")

board = Board(log_folder=log_folder)

MultilabelLogger.log(X, y1, y2, log_folder=log_folder)
board.serve(port=62010)
