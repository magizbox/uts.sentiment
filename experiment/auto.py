import json
from os.path import join, dirname
from underthesea.util.file_io import read
from experiment.aspect_svm_ngrams.model import aspect
from experiment.polarity_svm_ngrams.model import polarity


def auto_sentiment(text):
    sentiments = [{"aspect": aspect(text), "polarity": polarity(text)}]
    return sentiments


if __name__ == '__main__':
    data_file = join(dirname(dirname(__file__)), "data", "documents", "sent_20171125.json")
    data = read(data_file)
    posts = json.loads(data)

    for post in posts:
        text = post["text"]
        auto_sentiment(text)

