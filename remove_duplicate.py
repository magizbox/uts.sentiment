import unittest
from sklearn.base import BaseEstimator, TransformerMixin
import re


class RemoveDuplicate(BaseEstimator, TransformerMixin):
    def transform(self, x):
        result = []
        for s in x:
            s = re.sub(r'([a-z])\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
            s = re.sub(r'([a-z][a-z])\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
            result.append(s)
        return result

    def fit(self,x, y=None):
        return self


class TestRemoveDuplicate(unittest.TestCase):
    def test_duplicate_one_word(self):
        text = ['Chất liệu và kiểu dáng thì được nhưng quần hơi ngắn', 'Tôi thích lắm ahhhhh']
        expected = ['Chất liệu và kiểu dáng thì được nhưng quần hơi ngắn', 'Tôi thích lắm ah']
        actual = RemoveDuplicate().fit_transform(text)
        self.assertEqual(actual, expected)

    def test_duplicate_two_word(self):
        text = ['Tôi thích lắm ahhhhhhahhh', 'Tôi thích lắm ahahahah']
        expected = ['Tôi thích lắm ah', 'Tôi thích lắm ah']
        actual = RemoveDuplicate().fit_transform(text)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()