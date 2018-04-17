# [VLSP 2018 - Aspect Based Sentiment Analysis](http://vlsp.org.vn/vlsp2018/eval/sa)

## Cấu trúc

```
sentiment/
  ├─ data/                                  <!-- chứa dữ liệu trong quá trình huấn luyện và đánh giá mô hình 
  |   └─── vlsp2018/                        <!-- dữ liệu từ vlsp 2018
  |         ├── corpus/   
  |         │   ├── hotel                   <!-- dữ liệu khách sạn cho huấn luyện và đánh giá mô hình
  |         │   └── restaurant               <!-- dữ liệu nhà hàng
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

**Bước 1**: Chuẩn bị dữ liệu

- Dữ liệu gốc đặt tại thư mục `raw` gồm bộ dữ liệu khách sạn và nhà hàng, trong đó dữ liệu `train` và `dev` gồm nhận xét và nhãn tương ứng, ví dụ:
   ```
    Rộng rãi KS mới nhưng rất vắng. Các dịch vụ chất lượng chưa cao và thiếu.
    {HOTEL#DESIGN&FEATURES, positive}, {HOTEL#GENERAL, negative}
   ```
   dữ liệu `test` chỉ bao gồm các nhận xét.

- **Chạy file `preprocess.py`** để xử lí dữ liệu và đưa vào corpus. Quá trình xử lí dữ liệu gồm các bước:
   
   - Gộp 2 nhãn aspect và polarity, ngăn cách chúng bởi dấu `#`
  , ví dụ: `{HOTEL#DESIGN&FEATURES, positive}` --> `HOTEL#DESIGN&FEATURES#POSITIVE`
  - Ghi vào file excel nội dung gồm các nhận xét và nhãn tương ứng của nhận xét đó dưới dạng confusion matrix.
 - Sau quá trình xử lí dữ liệu, **chạy file `eda.py`**, module có nhiệm vụ mô tả về các cặp aspect + polarity. Kết quả của quá trình này thu được các hình ảnh lưu tại thư mục `eda`.

**Bước 2**: Huấn luyện mô hình

Chia làm 2 quá trình:
- Quá trình 1: **Chạy các file có tiền tố `turning`**: `turning_svc.py`, `turning_linearsvc`, ... nhằm huấn luyện dữ liệu với nhiều thử nghiệm (transfomer kết hợp estimator). Kết qủa F1 của các thử nghiệm lưu tại thư mục `logs`. Dữ liệu sử dụng là tại file `train.xlsx` hoặc kết hợp của 2 file `train.xlsx` và `dev.xlsx`. 

- Quá trình 2: Tìm kiếm các kết qủa F1 cao nhất từ quá trình 1, trích rút giá trị transfomer + estimator. Tiến hành huấn luyện lại dữ liệu bằng cách **chạy các file có tiền tố `train`**, các file `train` này có nhiệm vụ huấn luyện dữ liệu với transfomer + estimator nhận được, kết thúc quá trình huấn luyện nhận được các giá trị F1 đồng thời **trích xuất** các file `.bin` lưu trữ tại các thư mục trong `exported`. Tiến hành huấn luyện với dữ liệu trong `train.xlsx` hoặc dữ liệu đầy đủ (kết hợp dữ liệu `train.xlsx` + `dev.xlsx`)

**Bước 3**: Áp dụng mô hình

Từ các model tìm được trong quá trình 2 của bước 2, tiến hành gán nhãn dữ liệu. **Chạy xác file có tiền tố `make_result`**, dữ liệu biến đổi như trong ví dụ: `HOTEL#DESIGN&FEATURES#POSITIVE` --> `{HOTEL#DESIGN&FEATURES, positive}` và nhận được kết quả là các file lưu trong thư mục `results` với các nhận xét được gán nhãn.
