from model import sentiment
from unittest import TestCase


class TestSentiment(TestCase):
    def test_sentiment(self):
        text = "Gọi mấy lần mà lúc nào cũng là các chuyên viên đang bận hết ạ "
        actual = sentiment(text)
        expected = "CUSTOMER SUPPORT#NEGATIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_1(self):
        text = "Chúc mừng VCB, luôn thành công và tạo niềm tin cho khách hàng an tâm khi đồng hành cùng VCB."
        actual = sentiment(text)
        expected = "TRADEMARK#POSITIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_2(self):
        text = "Bạn ra ngân hàng hỏi luôn cho nhanh giải đáp qua đây ko ăn thua lắm."
        actual = sentiment(text)
        expected = "CUSTOMER SUPPORT#NEGATIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_3(self):
        text = "Tháng này tiền banking bị thu phí 3 lần liên tục đề nghị vietcombank xem xét lại "
        actual = sentiment(text)
        expected = "INTEREST RATE#NEGATIVE"
        self.assertEquals(actual[0], expected)

