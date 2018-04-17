from os.path import dirname, join
import re
from languageflow.util.file_io import read
import pandas as pd


def transform_comment(item):
    comment = {}
    comment["text"] = item.split("\n")[1]
    return comment


def convert_to_corpus(comments, corpus_file):
    df = pd.DataFrame(comments)
    df.to_excel(corpus_file, index=False)


if __name__ == '__main__':
    # =========================================================================#
    #                                 HOTEL                                    #
    # =========================================================================#
    data = read(join(dirname(__file__), "raw", "hotel", "3-VLSP2018-SA-Hotel-test (8-3-2018).txt")).split("\n\n")
    corpus_file = join(dirname(__file__), "corpus", "hotel", "test.xlsx")
    comments = [transform_comment(item) for item in data]
    convert_to_corpus(comments, corpus_file)

    # =========================================================================#
    #                               RESTAURANT                                 #
    # =========================================================================#
    data = read(join(dirname(__file__), "raw", "restaurant", "3-VLSP2018-SA-Restaurant-test (8-3-2018).txt")).split("\n\n")
    corpus_file = join(dirname(__file__), "corpus", "restaurant", "test.xlsx")
    comments = [transform_comment(item) for item in data]
    convert_to_corpus(comments, corpus_file)
