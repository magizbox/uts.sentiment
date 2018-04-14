# [VLSP 2018 - Aspect Based Sentiment Analysis](http://vlsp.org.vn/vlsp2018/eval/sa#data)

## Cấu trúc

```
sentiment/
  ├─ data/                                  <!-- chứa dữ liệu trong quá trình huấn luyện và đánh giá mô hình 
  |   └─── vlsp2018/                        <!-- dữ liệu từ vlsp 2018
  |         ├── corpus/   
  |         │   ├── hotel                   <!-- dữ liệu khách sạn huấn luyện và đánh giá mô hình
  |         │   └── restaurant              
  |         ├── raw/                        
  |         │   ├── hotel
  |         │   └── restaurant 
  |         ├── eda/
  |         ├── eda.py           
  |         └── preprocess.py               <!-- các bước tiền xử lí dữ liệu đưa vào corpus
  |
  └─ experiments/                           <!-- chứa các thí nghiệm với feature và model.
      ├── hotel/                             
      |     ├── exported/                   <!-- trích xuất model với các thử nghiệm tương ứng
      |     ├── logs/                       <!-- các kết quả thử nghiệm
      |     ├── results/                    <!-- dữ liệu được gán nhãn bằng máy
      |     ├── load_data.py                
      |     ├── model.py                    
      |     ├── make_results.txt                
      |     ├── score.py                    
      |     ├── test_model.py               
      |     ├── train.py                    
      |     └── turning.py                  
      |
      └── restaurant/
            ├── exported/                   <!-- trích xuất model với các thử nghiệm tương ứng
            ├── logs/                       <!-- các kết quả thử nghiệm
            ├── results/                    <!-- dữ liệu được gán nhãn bằng máy 
            ├── load_data.py                
            ├── model.py                    
            ├── make_results.txt                
            ├── score.py                    
            ├── test_model.py               
            ├── train.py                    
            └── turning.py  
```
## Xây dựng mô hình

**Bước 1**: Xử lí dữ liệu  

- Dữ liệu gốc đặt tại thư mục `raw` gồm bộ dữ liệu khách sạn và nhà hàng, trong đó dữ liệu `train` và `dev` gồm nhận xét và nhãn tương ứng, ví dụ:
```
Rộng rãi KS mới nhưng rất vắng. Các dịch vụ chất lượng chưa cao và thiếu.
{HOTEL#DESIGN&FEATURES, positive}, {HOTEL#GENERAL, negative}
```
dữ liệu `test` chỉ bao gồm các nhận xét.

- Quá trình xử lí dữ liệu gồm các bước:
   
   - Gộp 2 nhãn aspect và polarity, ngăn cách chúng bởi dấu `#`
  , ví dụ: `{HOTEL#DESIGN&FEATURES, positive}` --> `HOTEL#DESIGN&FEATURES#POSITIVE`
  - Ghi vào file excel nội dung gồm các nhận xét và nhãn tương ứng của nhận xét đó dưới dạng confusion matrix.

**Bước 2**: Huấn luyện dữ liệu

Chia làm 2 quá trình:
- Quá trình 1: Huấn luyện dữ liệu với nhiều thử nghiệm: nhiều transfomer kết hợp nhiều estimator. Kết qủa F1 của các thử nghiệm lưu tại thư mục `logs`. Dữ liệu sử dụng là kết hợp của 2 file `train.txt` và `dev.txt`

- Quá trình 2: Tìm kiếm các F1 cao nhất từ quá trình 1. Tiến hành huấn luyện lại dữ liệu với transfomer kết hợp estimator cho kết quả cao nhất đó nhằm trích xuất các file `.bin` lưu trữ tại các thư mục trong `exported`. Tiến hành huấn luyện với dữ liệu trong `train.txt` và dữ liệu đầy đủ (kết hợp dữ liệu `train.txt` + `dev.txt`)

**Bước 3**: Gán nhãn dữ liệu

Từ các model tìm được trong quá trinh 2 của bước 2, tiến hành gán nhãn dữ liệu. Các file `make_result` thực hiện bước này, dữ liệu biến đổi như trong ví dụ: `HOTEL#DESIGN&FEATURES#POSITIVE` --> `{HOTEL#DESIGN&FEATURES, positive}`
