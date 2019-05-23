# Hướng dẫn import dữ liệu 

## Dữ liệu VLSP2016_SA

Bộ dữ liệu VLSP2016_SA có thể được import với công cụ langaugeflow

Cài đặt languageflow

```
pip install languageflow==1.1.13a0 
```

Gửi yêu cầu dữ liệu đến nhóm tác giả của dữ liệu VLSP2016_SA, sau khi nhận được dữ liệu, hãy chắc chắn cấu trúc thư mục của dữ liệu như sau 

```
vlsp2016_sa_raw/
├── SA2016-TestData-Ans
│   ├── test_raw_ANS.txt
│   └── test_tokenized_ANS.txt
├── SA2016-training_data
│   ├── SA-training_negative.txt
│   ├── SA-training_neutral.txt
│   └── SA-training_positive.txt
└── SA2016-training-data-ws
    ├── train_negative_tokenized.txt
    ├── train_neutral_tokenized.txt
    └── train_positive_tokenized.txt
```

Import dữ liệu vào languageflow

```
languageflow import vlsp2016_sa_raw 
```

Kiểm tra dữ liệu

```
$ python 

>>> from languageflow.data import CategorizedCorpus
>>> from languageflow.data_fetcher import DataFetcher, NLPData
>>> corpus = DataFetcher.load_corpus(NLPData.VLSP2016_SA)
>>> corpus.train[:5]
[
  Sentence: " Đang điên vì đung android hỗ trợ phần mềm cùi quá... ios ít khi bị dis,android bị hoài. Mong là sẽ có update ngon hơn. Chứ thấy trừ nhịp tim,còn lại thua i5+,. Bùn ghê...  " - Labels: [NEG (1.0)],
  Sentence: " rồi khi có người gọi tới sim đt gắn trên smp thì lúc đó nghe bằng niềm tin hả, sim trên sw là số khác nhé, còn nếu dùng dịch vụ chuyển cuộc gọi thì khi nghe cũng bị tính tiền như khi gọi nhé, chơi chuyển hướng một ngày nhận chục cuộc gọi thôi là tháy tiền ra đi vì đâu rồi, sao còn í định dùng sw có sim nữa không  " - Labels: [NEG (1.0)],
  Sentence: " Thich Sam sung dac biet la note tu lau roi" - Labels: [POS (1.0)]
  Sentence: " Mình cũng thích màn hình thẳng hơn và thích pin rời dù biết pin gắn trong sẽ làm máy đẹp hơn" - Labels: [NEU (1.0)]
  Sentence: " Quá tuyệt vời cho samsung, xứng đáng là hãng công nghệ đứng số 1 toàn cầu, bỏ xa các đối thủ khác, iphone thì ko chấp, thua cả xiaomi của trung quốc." - Labels: [POS (1.0)]
]

```
