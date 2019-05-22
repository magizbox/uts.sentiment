# Nhận diện cảm xúc tiếng Việt

![](https://img.shields.io/badge/made%20with-%E2%9D%A4-red.svg)
![](https://img.shields.io/badge/opensource-vietnamese-blue.svg)
![](https://img.shields.io/badge/build-passing-green.svg)

Dự án nghiên cứu về bài toán *nhận diện cảm xúc tiếng Việt*, được phát triển bởi nhóm nghiên cứu xử lý ngôn ngữ tự nhiên tiếng Việt - [underthesea](https://github.com/undertheseanlp). Chứa mã nguồn các thử nghiệm cho việc xử lý dữ liệu, huấn luyện và đánh giá mô hình, cũng như cho phép dễ dàng tùy chỉnh mô hình đối với những tập dữ liệu mới.

**Nhóm tác giả** 

* Vũ Anh <<anhv.ict91@gmail.com>>
* Phạm Hồng Quang <<quangphampm@gmail.com>>
* Phạm Thị Ánh Sương  <<phamsuong1551997@gmail.com>>

[Danh sách những người đóng góp](AUTHORS.md) 

**Tham gia đóng góp**

 Mọi ý kiến đóng góp hoặc yêu cầu trợ giúp xin gửi vào mục [Issues](../../issues) của dự án. Các thảo luận được khuyến khích **sử dụng tiếng Việt** để dễ dàng trong quá trình trao đổi. 
 
Nếu bạn có kinh nghiệm trong bài toán này, muốn tham gia vào nhóm phát triển với vai trò là [Developer](https://github.com/undertheseanlp/underthesea/wiki/H%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-%C4%91%C3%B3ng-g%C3%B3p#developercontributor), xin hãy đọc kỹ [Hướng dẫn tham gia đóng góp](https://github.com/undertheseanlp/underthesea/wiki/H%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-%C4%91%C3%B3ng-g%C3%B3p#developercontributor).


## Mục lục

* [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
* [Thiết lập môi trường](#thiết-lập-môi-trường)
* [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
* [Kết quả thử nghiệm](#kết-quả-thử-nghiệm)
* [Trích dẫn](#trích-dẫn)



## Yêu cầu hệ thống 

* `Hệ điều hành: Linux (Ubuntu, CentOS), Mac`
* `Python 3.6+`
* `conda 4+`

## Thiết lập môi trường

Tải project bằng cách sử dụng lệnh `git clone`

```
$ git clone https://github.com/undertheseanlp/sentiment.git
```

Tạo môi trường mới và cài đặt các gói liên quan

```
$ cd sentiment
$ conda create -n sentiment python=3.6
$ pip install -r requirements.txt
```

## Hướng dẫn sử dụng

Trước khi chạy các thử nghiệm, hãy chắc chắn bạn đã activate môi trường `sentiment`, mọi câu lệnh đều được chạy trong thư mục gốc của dự án.

```
$ cd sentiment
$ source activate sentiment
```

## Kết quả thử nghiệm 

Xem thêm về [*mô tả vlsp 2018 SA task*](http://vlsp.org.vn/vlsp2018/eval/sa)

Kết quả các thử nghiệm

**Dữ liệu restaurant**

| Mô hình                                                                         | F1 %     |
|---------------------------------------------------------------------------------|----------|
| XGBoost(n_iter=500, max_depth=500) + Countvectorizer(bigram, max_features=4000) | **65.55** |
| LogisticRegression + Countvectorizer(Bigram)                                    | 64.59     |
| LinearSVC + Countvectorizer(Trigram)                                            | 64.49     |
| LinearSVC + Countvectorizer(Bigram)                                             | 64.24     |
| MultinomialNB + Countvectorizer(Trigram)                                        | 53.66     |
| SVC + Countvectorizer(Trigram)                                                  | 48.55     |

**Dữ liệu hotel**

| Mô hình                                                                         | F1 %      |
|---------------------------------------------------------------------------------|-----------|
| XGBoost(n_iter=100, max_depth=200) + Countvectorizer(bigram, max_features=2000) | **65.79** |
| LinearSVC + Countvectorizer(Trigram)                                            | 65.09     |
| LinearSVC + Countvectorizer(Bigram)                                             | 64.95     |
| LogisticRegression + Countvectorizer(Bigram)                                    | 64.82     |
| MultinomialNB + Countvectorizer(Bigram)                                         | 54.79     |
| SVC + Countvectorizer(Trigram)                                                  | 48.55     |

**Dữ liệu fb_bank**

| Mô hình                                                                         | F1 %     |
|---------------------------------------------------------------------------------|----------|
| LinearSVC + Tfidfvectorizer(Bigram)                                             | **69.60**|
| MultinomialNB + Countvectorizer (Bigram, Max Feature=1000)                      | 68.40    |
| XGBoost(n_iter=100, max_depth=300) + Countvectorizer(bigram, max_features=2000) | 65.70    |
| LogisticRegression + Countvectorizer(Trigram, max_features=5000)                | 65.70    |
| SVC + Countvectorizer(Trigram, Max Feature=700)                                 | 29.60    |

## Cài đặt sacred

Cài đặt `sacred` python package

```
pip install sacred 
```

Cài đặt database

```
sudo apt get install docker.io
sudo docker run -p 27017:27017 mongo
```

Cài đặt omniboard 

```
sudo npm install -g omniboard
omniboard
```

Sau đó vào địa chỉ [http://localhost:9000](http://localhost:9000) để xem kết quả các thử nghiệm 