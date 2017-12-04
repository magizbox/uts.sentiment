from unittest import TestCase


class TestAutoSentiment(TestCase):
    def test_auto_sentiment(self):
        text = ""
        actual = auto(text)