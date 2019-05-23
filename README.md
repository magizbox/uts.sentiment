# Nhận diện cảm xúc tiếng Việt

![](https://img.shields.io/badge/made%20with-%E2%9D%A4-red.svg)
![](https://img.shields.io/badge/opensource-vietnamese-blue.svg)
![](https://img.shields.io/badge/build-passing-green.svg)

Dự án nghiên cứu về bài toán *nhận diện cảm xúc tiếng Việt*, được phát triển bởi nhóm nghiên cứu xử lý ngôn ngữ tự nhiên tiếng Việt - [underthesea](https://github.com/undertheseanlp). Chứa mã nguồn các thử nghiệm cho việc xử lý dữ liệu, huấn luyện và đánh giá mô hình, cũng như cho phép dễ dàng tùy chỉnh mô hình đối với những tập dữ liệu mới.

## Kết quả thử nghiệm 

| Dữ liệu          | Mô hình                                             | F1 %     |
|-----------------|------------------------------------------------------|----------|
| VLSP2018_SA_RES | XGBoost + Countvectorizer(bigram, max_features=4000) | 65.55    |
| VLSP2016_SA_HOT | XGBoost + Countvectorizer(bigram, max_features=2000) | 65.79    |
| UTS2017_BANK    | LinearSVC + Tfidfvectorizer(Bigram)                  | 69.60    |
| VLSP2016_SA     | SVM + TfidfVectorizer                                | 70.02    |

## Hướng dẫn sử dụng nhanh

**Yêu cầu hệ thống**

* `Hệ điều hành: Linux (Ubuntu, CentOS), Mac`
* `Python 3.6+`, `conda`

**Cài đặt** 

```
# Tải project bằng cách sử dụng lệnh `git clone`
$ git clone https://github.com/undertheseanlp/sentiment.git

# Tạo môi trường mới và cài đặt các gói liên quan
$ cd sentiment
$ conda create -n sentiment python=3.6
$ pip install -r requirements.txt 
```

### Huấn luyện mô hình sentiment analysis cho dữ liệu VLSP2016_SA

**Chú ý: Xem [hướng dẫn import dữ liệu](docs/TUTORIAL_1_IMPORT_DATA.md) trước khi huấn luyện mô hình**  

```
$ cd sentiment
$ source activate sentiment

# Sử dụng mô hình để predict 
python vlsp2016_predict.py
```

Huấn luyện mô hình

```
$ python vlsp2016_train.py
```
 
Sử dụng mô hình đã huấn luyện để dự đoán 

```
$ python vlsp2016_predict.py

Load model from tmp/sentiment_svm_vlsp2016
Model is loaded.

Text: Sản phẩm rất tốt
Labels: ['POS']

Text: Pin yếu quá
Labels: ['NEG']
```

## Hướng dẫn sử dụng chi tiết 

* [Hướng dẫn 1: Import dữ liệu vào hệ thống](docs/TUTORIAL_1_IMPORT_DATA.md)
* [Hướng dẫn 2: Cách huấn luyện mô hình](docs/TUTORIAL_2_TRAINING_MODELS.md)
* [Hướng dẫn 3: Các tối ưu mô hình](docs/TUTORIAL_3_OPTIMIZE_MODELS.md)

## Trích dẫn undertheseanlp@sentiment

```
@online{undertheseanlp/sentiment,
author ={Vu Anh, Pham Hong Quang, Pham Thi Anh Suong},
year = {2019},
title ={Phân loại văn bản tiếng Việt},
url ={https://github.com/undertheseanlp/classification}
}
```

