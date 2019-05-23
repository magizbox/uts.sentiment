from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier

def predict(text):
    print(f"\nText: {text}")
    model_folder = "tmp/sentiment_svm_vlsp2016"
    classifier = TextClassifier.load(model_folder)
    sentence = Sentence(text)
    classifier.predict(sentence)
    labels = sentence.labels
    print(f"Labels: {labels}")

predict('Sản phẩm rất tốt')
predict('Dịch vụ rất chán')