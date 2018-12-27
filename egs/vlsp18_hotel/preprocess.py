from os.path import dirname, join
import pandas as pd
import re


def read(path):
    with open(path, encoding='utf-8') as f:
        content = f.read().split("\n\n")
    return content


def transform(s):
    sentence = {}
    sentence["text"] = s.split("\n")[1]
    sentiments = s.split("\n")[2]
    sentiments_ = re.split("}, +{", sentiments)
    sentiments__ = [re.sub(r"[{}]", "", item) for item in sentiments_]
    labels = [item.upper().replace(", ", "#") for item in sentiments__]
    sentence["labels"] = labels
    return sentence


def convert_to_corpus(sentences, file_path):
    data = []
    labels = list(set(sum([s["labels"] for s in sentences], [])))
    for s in sentences:
        item = {}
        item["text"] = s["text"]
        for label in labels:
            if label in s["labels"]:
                item[label] = 1
            else:
                item[label] = 0
        data.append(item)
    df = pd.DataFrame(data)
    columns = ["text"] + labels
    df.to_excel(file_path, index=False, columns=columns)


if __name__ == '__main__':
    path = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "hotel")
    corpus = join(dirname(__file__), "data")
    train_data = read(join(path, "1-VLSP2018-SA-hotel-train (7-3-2018).txt"))
    train_data = [transform(sent) for sent in train_data]
    convert_to_corpus(train_data, join(corpus, "train.xlsx"))

    dev_data = read(join(path, "2-VLSP2018-SA-hotel-dev (7-3-2018).txt"))
    dev_data = [transform(sent) for sent in dev_data]
    convert_to_corpus(dev_data, join(corpus, "dev.xlsx"))
