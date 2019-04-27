from enum import Enum
from pathlib import Path
import re

import click


def preprocess_aivivn2019_sa():
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


def preprocess_vlsp2016_sa(raw_data_folder, normalized_data_folder):
    # Preprocess Training Data
    def process_train_data(train_data_file, output_file, label):
        text = open(train_data_file).read()
        text = text.strip()
        sentences = text.split("\n\n")
        with open(output_file, "a") as f:
            for s in sentences:
                f.write(f"__label__{label} {s}\n")

    train_data_folder = raw_data_folder / "SA2016-TrainingData"
    output_train_file = normalized_data_folder / "train.txt"
    with open(output_train_file, "w") as f:
        f.write("")
    process_train_data(train_data_folder / "SA-training_positive.txt", output_train_file, "POS")
    process_train_data(train_data_folder / "SA-training_neutral.txt", output_train_file, "NEU")
    process_train_data(train_data_folder / "SA-training_negative.txt", output_train_file, "NEG")

    # Preprocess Test Data
    test_sentences = []
    with open(raw_data_folder / "SA2016-TestData-Ans" / "test_raw_ANS.txt", "r") as f:

        for i, line in enumerate(f):
            if i % 2 == 0:
                text = line.strip()
            else:
                label = line.strip()
                sentence = f"__label__{label} {text}"
                test_sentences.append(sentence)
    output_test_file = normalized_data_folder / "test.txt"
    with open(output_test_file, "w") as f:
        content = "\n".join(test_sentences)
        f.write(content + "\n")


class Dataset(Enum):
    AIVIVN2019_SA = "AIVIVN2019_SA"
    VLSP2016_SA = "VLSP2016_SA"


CACHE_ROOT = Path("tmp")


@click.command()
@click.argument("dataset", required=True)
@click.argument("raw_data_folder", required=False)
@click.argument("normalized_data_folder", required=False)
def main(dataset, raw_data_folder, normalized_data_folder):
    datasets = Dataset._value2member_map_
    dataset_name = dataset
    if dataset_name not in datasets:
        print("Please check dataset name. ")
        return

    dataset = datasets[dataset_name]
    raw_data_folder = CACHE_ROOT / dataset.value / "raw"
    normalized_data_folder = CACHE_ROOT / dataset.value / "normalized"

    if dataset == Dataset.VLSP2016_SA:
        preprocess_vlsp2016_sa(raw_data_folder, normalized_data_folder)

    if dataset == Dataset.AIVIVN2019_SA:
        preprocess_aivivn2019_sa()


if __name__ == '__main__':
    main()
