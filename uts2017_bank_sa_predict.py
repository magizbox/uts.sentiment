from languageflow.data import Sentence
from languageflow.data_fetcher import NLPData, DataFetcher
from languageflow.models.text_classifier import TextClassifier

model_folder = "tmp/uts2017_bank_sa"
print(f"Load model from {model_folder}")
classifier = TextClassifier.load(model_folder)
print(f"Model is loaded.")


def predict(text):
    print(f"\nText: {text}")

    sentence = Sentence(text)
    classifier.predict(sentence)
    labels = sentence.labels
    print(f"Labels: {labels}")

corpus = DataFetcher.load_corpus(NLPData.UTS2017_BANK_SA)

predict('Bạn nên làm thẻ credit, đừng làm debit. Mình dùng thẻ debit của vcb, tết vừa rồi bị hack mất 28 triệu trong tài khoản mà đến giờ vcb đã giải quyết cho mình đâu. Bực mình!')
