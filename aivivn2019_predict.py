from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier

model_folder = 'tmp/sentiment_svm_aivivn2019'
print(f"Load model from {model_folder}")
classifier = TextClassifier.load(model_folder)
print(f"Model is loaded.")


def predict(text):
    print(f"\nText: {text}")

    sentence = Sentence(text)
    classifier.predict(sentence)
    labels = sentence.labels
    print(f"Labels: {labels}")


predict("hàng kém chất lg,chăn đắp lên dính lông lá khắp người. thất vọng")
predict("Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt, đóng gói cẩn thận.")