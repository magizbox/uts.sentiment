from unittest import TestCase

from experiment.aspect_svm_ngrams.model import aspect


class TestCategory(TestCase):
    def test_category(self):
        text = "Mở tài khoản ATM thì có đc quà ko ad"
        actual = aspect(text)
        expected = "ACCOUNT"
        self.assertEquals(expected, actual[0])

    def test_category_2(self):
        text = "Cần tư vấn mà add k rep "
        actual = aspect(text)
        expected = "CUSTOMER SUPPORT"
        self.assertEquals(expected, actual[0])

    def test_category_3(self):
        text = "Mình đã đăng nhập thành công. Nhưng ko sử dụng đc các dich vụ. Kể cả xe số dư hay chuyển tiền "
        actual = aspect(text)
        expected = "INTERNET BANKING"
        self.assertEquals(expected, actual[0])
