from pathlib import Path
import re
corpus_folder = Path("tmp/AIVIVN2019_SA/")
train_file = open(corpus_folder / "normalized/train.txt", "w")
dev_file = open(corpus_folder / "normalized/dev.txt", "w")
test_file = open(corpus_folder / "normalized/test.txt", "w")

content = open(corpus_folder / "raw/train.crash").read()
content = content.strip().replace(u'\xa0', u' ')
sections = re.split("train_", content)[1:]
sentences = []
for i, section in enumerate(sections):
    lines = section.strip().split("\n")
    sentence_id = int(lines[0])
    text = " ".join(lines[1:-1])
    text = text.replace("\n", " ")
    label = lines[-1]
    sentences.append((sentence_id, text, label))
    line_content = f"__label__{label} {text}\n"
    train_file.write(line_content)

def load_labels(file):
    result = {}
    lines = open(file).read().splitlines()[1:]
    for line in lines:
        id, label = line.split(",")
        id = int(id[5:])
        result[id] = label
    return result

public_file_path = corpus_folder / "raw/public_test_label.csv"
dev = load_labels(public_file_path)
private_file_path = corpus_folder / "raw/private_test_label.csv"
test = load_labels(private_file_path)


content = open(corpus_folder / "raw/test.crash").read()
content = content.strip().replace(u'\xa0', u' ')
sections = re.split("test_", content)[1:]
for i, section in enumerate(sections):
    lines = section.strip().split("\n")
    s_id = int(lines[0])
    text = " ".join(lines[1:])
    text = text.replace("\n", " ")
    if s_id in dev:
        label = dev[s_id]
        f = dev_file
    if s_id in test:
        label = test[s_id]
        f = test_file
    line_content = f"__label__{label} {text}\n"
    f.write(line_content)
