from os.path import join, dirname
import re

from snapshots import sentiment


def read(path):
    with open(path, encoding='utf-8') as f:
        content = f.read().split("\n\n")
    content = [i.split('\n')[1] for i in content]
    return content


def generate_labels(y):
    labels = []
    for item in y:
        matched = re.match("^(?P<attribute>.*)#(?P<sentiment>\s*POSITIVE|NEGATIVE|NEUTRAL)$", item)
        attribute = matched.group("attribute")
        sentiment = matched.group("sentiment")
        label = "{}, {}".format(attribute, sentiment.lower())
        label = "{" + label + "}"
        labels.append(label)
    labels = ", ".join(labels)
    return labels


if __name__ == '__main__':
    test_path = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "restaurant",
                     "3-VLSP2018-SA-Restaurant-test (8-3-2018).txt")
    result_path = join(dirname(dirname(dirname(__file__))), "evaluate", "SA_Evaluate", "restaurant.txt")
    X_test = read(test_path)
    y = [sentiment(x) for x in X_test]
    content = ""
    for i in range(len(X_test)):
        content += "#{}\n".format(i + 1)
        content += "{}\n".format(X_test[i])
        content += "{}\n\n".format(generate_labels(y[i]))
    with open(result_path, "w") as f:
        f.write(content)
