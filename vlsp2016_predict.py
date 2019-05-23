from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier

model_folder = "tmp/sentiment_svm_vlsp2016"
print(f"Load model from {model_folder}")
classifier = TextClassifier.load(model_folder)
print(f"Model is loaded.")


def predict(text):
    print(f"\nText: {text}")

    sentence = Sentence(text)
    classifier.predict(sentence)
    labels = sentence.labels
    print(f"Labels: {labels}")


predict('Sản phẩm rất tốt')
predict('Pin yếu quá')
