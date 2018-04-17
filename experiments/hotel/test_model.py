from unittest import TestCase

from svc_exported import sentiment


class TestSentiment(TestCase):
    def test_sentiment_1(self):
        text = "Chưa có thang máy. Chưa chấp nhận thanh toán bằng thẻ. Địa điểm dễ tìm, bày trí bằng tre nứa rất mát mẻ, bạn lễ tân nhiệt tình, niềm nở, thân thiện, tốt bụng cực kỳ. Tôi đặt phòng 1 giường đôi nhưng biết tôi đi cùng 2 con nhỏ nên ks đã chủ động chuẩn bị thêm 1 chiếc giường tầng cho 2 bé. Phòng rộng rãi, bày trí đẹp, giường êm, nhà vệ sinh cực sạch sẽ. Vị trí thuận lợi, dễ di chuyển. Bạn lễ tân đã giới thiệu tôi các điểm ăn ngon trên đường Lê Thị Riêng, Nguyễn Trãi gần đó. Tóm lại, Nguyễn Shack xứng đáng để tôi giới thiệu bạn bè."
        actual = sentiment(text)
        expected = ('AMBIENCE#GENERAL#NEUTRAL', 'FOOD#PRICE#POSITIVE', 'FOOD#QUALITY#POSITIVE', 'SERVICE#GENERAL#NEUTRAL')
        self.assertEquals(actual, expected)

    def test_sentiment_2(self):
        text = "Bánh ăn thơm và vừa miệng. Gìon rụm, ngọt nhưng không quá gắt. Đáng để thử "
        actual = sentiment(text)
        expected = ('FOOD#QUALITY#POSITIVE',)
        self.assertEquals(actual, expected)

    def test_sentiment_3(self):
        text = "ấn tg đầu tiên, quá đắt,50k cho 1 xuất cơm ntn. Là m lâu k ăn cơm, k biết giá hay cơm đắt thật k biết " \
               "nữa. Cơm chả có j đặc biệt :)k bh ăn nữa "
        actual = sentiment(text)
        expected = ('FOOD#PRICE#NEGATIVE', 'FOOD#QUALITY#POSITIVE')
        self.assertEquals(actual, expected)

    def test_sentiment_4(self):
        text = "Ăn bình thường, cho nhiều hỗn hợp quá nên ăn mau ngán. Theo mình là khá ngọt nữa. 1 đĩa chỉ có 25k " \
               "nên ko đòi hỏi gì nhiều. Mình thấy mấy phần fast food ăn ok hơn nhiều nha. 2 người tầm 80k là đủ no " \
               "rồi. "
        actual = sentiment(text)
        expected = ('FOOD#PRICE#NEUTRAL', 'FOOD#QUALITY#POSITIVE')
        self.assertEquals(actual, expected)