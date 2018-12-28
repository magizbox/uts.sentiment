# Phân tích cảm xúc tiếng Việt

![](https://img.shields.io/badge/made%20with-%E2%9D%A4-red.svg)
![](https://img.shields.io/badge/opensource-vietnamese-blue.svg)
![](https://img.shields.io/badge/contributions-welcome-green.svg)


Dự án nghiên cứu về bài toán *phân tích cảm xúc tiếng Việt*, được phát triển bởi nhóm nghiên cứu xử lý ngôn ngữ tự nhiên tiếng Việt - [underthesea](https://github.com/undertheseanlp). Chứa mã nguồn các thử nghiệm cho việc xử lý dữ liệu, huấn luyện và đánh giá mô hình, cũng như cho phép dễ dàng tùy chỉnh mô hình đối với những tập dữ liệu mới.


**Nhóm tác giả** 

* Bùi Nhật Anh ([buinhatanh1208@gmail.com](buinhatanh1208@gmail.com))
* Vũ Anh ([anhv.ict91@gmail.com](anhv.ict91@gmail.com))


**Tham gia đóng góp**

Mọi ý kiến đóng góp hoặc yêu cầu trợ giúp xin gửi vào mục [Issues](../../issues) của dự án. Các thảo luận được khuyến khích **sử dụng tiếng Việt** để dễ dàng trong quá trình trao đổi. 

Nếu bạn có kinh nghiệm trong bài toán này, muốn tham gia vào nhóm phát triển với vai trò là [Developer](https://github.com/undertheseanlp/underthesea/wiki/H%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-%C4%91%C3%B3ng-g%C3%B3p#developercontributor), xin hãy đọc kỹ [Hướng dẫn tham gia đóng góp](https://github.com/undertheseanlp/underthesea/wiki/H%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-%C4%91%C3%B3ng-g%C3%B3p#developercontributor).

# Kết quả thử nghiệm

| Model                                                          | F1 Score (%) |
|----------------------------------------------------------------|--------------|
| LinearSVC (Tfidf_ngrams(1,2) + max_features=5000)              | 63.5         |
| LinearSVC (Tfidf_ngrams(1,2)                                   | 63.0         |
| SVC (Count_ngrams(1,2) + max_df=0.5 + min_df=8)                | 59.5         |
| Logistic Regression (Count_ngrams(1,2) + max_df=0.8+ min_df=8) | 58.3         |
| SVC (Count_ngrams(1,2) + max_df=0.8 + min_df=0.005)            | 57.6         |
| Logistic Regression (Tfidf_ngrams(1,2) + max_df=0.8+ min_df=8) | 53.6         |


## Bản quyền

Mã nguồn của dự án được phân phối theo giấy phép [GPL-3.0](LICENSE.txt).