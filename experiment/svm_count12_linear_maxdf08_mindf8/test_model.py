from model import sentiment
from unittest import TestCase


class TestSentiment(TestCase):
    def test_sentiment(self):
        text = "Gọi mấy lần mà lúc nào cũng là các chuyên viên đang bận hết ạ "
        actual = sentiment(text)
        expected = "CUSTOMER SUPPORT#NEGATIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_1(self):
        text = "Không tin tưởng vào ngân hàng BIDV "
        actual = sentiment(text)
        expected = "TRADEMARK#NEGATIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_2(self):
        text = "Bạn ra ngân hàng hỏi luôn cho nhanh giải đáp qua đây ko ăn thua lắm."
        actual = sentiment(text)
        expected = "CUSTOMER SUPPORT#NEGATIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_3(self):
        text = "VCB Mobile B@bking. Chuyển rất nhanh. Giao diện đẹp, thân thiện. Dễ thao tác"
        actual = sentiment(text)
        expected = "INTERNET BANKING#POSITIVE"
        self.assertEquals(actual[0], expected)

