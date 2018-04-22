# [VLSP 2018 - Aspect Based Sentiment Analysis](http://vlsp.org.vn/vlsp2018/eval/sa)

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Sentiment Analysis Project](#sentiment-analysis-project)
    * [Giới thiệu](#introduction)
	* [Cài đặt môi trường](#setup-environment)
	* [Tính năng](#features)
	* [Hướng dẫn sử dụng](#usage)
	* [Cấu trúc thư mục](#folder-structure)
	* [Chi tiết về dự án](#customization)
		* [Dữ liệu](#data)
		* [Huấn luyện mô hình](#trainer)
		* [Mô hình](#model)
		* [Chỉ số đánh giá](#evaluate)
		* [Ghi lại log](#logging)
		* [Áp dụng mô hình](#predict)

<!-- /code_chunk_output -->

## Giới thiệu

Dự án chứa các thử nghiệm trong các vấn đề phân tích cảm xúc câu với tiếng Việt. Đây là một phần của dự án [underthesea](https://github.com/magizbox/underthesea).


## Cài đặt môi trường
```
# clone project
$ git clone https://github.com/undertheseanlp/sentiment

# create environment
$ cd sentiment
$ conda create -n sentiment python=3.5
$ source activate sentiment
$ pip install -r requirements.txt
```

## Tính năng
* Giảm bớt các thư mục, thực hiện nhiều hơn các thử nghiệm.
* Thư mục `exported` chứa các model và features dạng file `.bin` được trích xuất từ quá trình huấn luyện mô hình để sử dụng cho việc triển khai mô hình trên các hệ thống khác.
* `logs` là kết quả của các quá trình thử nghiệm.
* `results` chứa dữ liệu của việc áp dụng mô hình với các dữ liệu cần gán nhãn.

## Hướng dẫn sử dụng
- Tại các thư mục các thử nghiệm, chạy các file có tiền tố `turning` bằng câu lệnh:
  ```
  python file_name.py
  ```
- Tiến hành huấn luyện mô hình. Tương tự, chạy các file có tiền tố  `train`.
- Tiếp theo, kiểm tra hoạt động của mô hình bằng cách chạy file `test_model.py`
- Từ các mô hình đã trích xuất, tiến hành áp dụng nó với dữ liệu cần gán nhãn. Chạy các file có tiền tố  `make_result`, kết quả thu được là các file text đã chứa nhận xét được gán nhãn.

## Cấu trúc thư mục
```
sentiment/
  ├─ data/                                  <!-- chứa dữ liệu trong quá trình huấn luyện và đánh giá mô hình 
  |   └─── fb_bank/                         <!-- nhận xét khách hàng với dịch vụ ngân hàng
  |   |     ├── corpus/   
  |   |     │   ├── data.xlsx  
  |   |     │   ├── test.xlsx 
  |   |     │   └── train.xlsx
  |   |     ├── raw/                        
  |   |     │   └── sentiments.json 
  |   |     ├── eda/
  |   |     ├── eda.py           
  |   |     └── preprocess.py               <!-- các bước tiền xử lí dữ liệu đưa vào corpus
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
      ├── bank/                             <!-- các thử nghiệm có cấu trúc giống nhau được đưa vào 1 thư mục khác nhau với tên của feature và model 
      |     └── thử nghiệm/                  
      |          ├── analyze/                       
      |          ├── model/                   
      |          ├── load_data.py                
      |          ├── model.py                    
      |          ├── test_model.py                
      |          ├── analyze.py                                      
      |          └── train.py    
      |
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

## Chi tiết về dự án
#### Dữ liệu
* **Chuẩn bị dữ liệu**
  1. **Dữ liệu gốc ```raw```**
     * Chứa bộ dữ liệu huấn luyện và kiểm tra.
     * Gồm các chủ đề khách sạn, nhà hàng, ngân hàng.
     * Dữ liệu là các nhận xét và nhãn tương ứng (khía cạnh#cảm xúc).

  2. **Dữ liệu`corpus`**

     Để tạo bộ dữ liệu `corpus` thực hiện module `preprocess`:
     * ```convert_to_corpus()```: biến đổi dữ liệu gốc gồm nhận xét và nhãn tương ứng của nhận xét đó dưới dạng confusion matrix, đồng thời lưu lại thành định dạng excel.
     * ```transform```: ghép 2 nhãn aspect và polarity từ nhãn đã gán thành dạng aspect#polarity 

* **Phân tích thăm dò**

  ```analyze``` tiến hành tính tổng các bộ dữ liệu `train`, `dev`, `test` và phân phối các nhãn trong dữ liệu đồng thời lưu hình ảnh tại thư mục `eda`:
  ```python
  	df = pd.read_excel(path, encoding='sys.getfilesystemencoding()')
  	print("\t- size:", df.shape)
    rcParams['figure.figsize'] = 30, 15
    df.drop("text", axis=1).sum().plot.barh()
  ```

#### Huấn luyện mô hình
  1. **Thực hiện nhiều thử nghiệm với ```turing```**
   * Lấy dữ liệu từ corpus.
   * Biến đổi dữ liệu với các đặc trưng `TfidfVectorizer` và `CountVectorizer`.
   * Huấn luyện mô hình với dữ liệu đã biến đổi bằng hàm `train`.
   * Kiểm tra mô hình, trích xuất chỉ số `F1` và ghi ra log lưu tại thư mục `logs`.

  2. **Trích xuất mô hình**

     Tìm kiếm tại thư mục `logs` kết quả tốt nhất ứng với `estimator` và `features`. Thực hiện huấn luyện lại mô hình với các bước trên, sau đó trích xuất mô hình bằng hàm `export`

#### Mô hình

Chứa các interface trong việc huấn luyện mô hình:
* ```get_name```: Lấy tên feature và estimator.
* ```load_data```: Lấy dữ liệu.
* ```fit_transform```: Biến đổi dữ liệu với các feature.
* ```train```: Huấn luyện mô hình.
* ```evaluate```: Đánh giá mô hình.

 Các thử nghiệm trong mô hình gồm: SVC, LinearSVC, Xgboost, Linear Regression, Naive Bayes kết hợp các đăc trưng `TfidfVectorizer` và `CountVectorizer` với các chỉ số `ngrams`, `max_features`.

#### Chỉ số đánh giá
Đánh giá mô hình với chỉ số f1 bằng hàm `multilabel_f1_score` trong file `score.py`.

#### Ghi lại log
Trong quá trình thử nghiệm, việc ghi lại kết quả f1 giúp tìm được thử nghiệm có kết quả tốt nhất, từ đó huấn luyện lại và trích xuất mô hình.

#### Áp dụng mô hình
Trong phạm vi project, chúng tôi tiến hành áp dụng mô hình cho việc gán nhãn dữ liệu cho SA task VLSP 2018. Quá trình gán nhãn gồm các bước:
   * Lấy dữ liệu từ corpus.
   * Gán nhãn dữ liệu là các câu nhận xét bằng hàm `sentiment`.
   * Biến đổi lại dữ liệu từ dạng tuple(aspect#polarity) về {aspect, polarity}.
   * Kết quả của quá trình lưu tại thư mục `result`.