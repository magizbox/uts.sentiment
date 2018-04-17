# [VLSP 2018 - Aspect Based Sentiment Analysis](http://vlsp.org.vn/vlsp2018/eval/sa)

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Sentiment Analysis Project](#sentiment-analysis-project)
    * [Giới thiệu](#introduction)
	* [Cài đặt môi trường](#setup-environment)
	* [Tính năng](#features)
	* [Hướng dẫn sử dụng](#usage)
	* [Cấu trúc thư mục](#folder-structure)
	* [Customization](#customization)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss & Metrics](#loss-metrics)
			* [Multiple metrics](#multiple-metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoint naming](#checkpoint-naming)
	* [Contributing](#contributing)
	* [TODOs](#todos)
	* [Acknowledgments](#acknowledgments)

<!-- /code_chunk_output -->

## Giới thiệu

Dự án chứa các thí nghiệm trong các vấn đề phân tích cảm xúc câu với tiếng Việt. Đây là một phần của dự án [underthesea](https://github.com/magizbox/underthesea).


## Cài đặt môi trường
```
# clone project
$ git clone https://github.com/undertheseanlp/sentiment

# create environment
$ cd sentiment
$ conda create -n sentiment python=3.5
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
- Tiếp theo kiểm tra hoạt động của mô hình bằng cách chạy file `test_model.py`
- Từ các mô hình đã trích xuất, tiến hành áp dụng nó với dữ liệu cần gán nhãn. Chạy các file có tiền tố  `make_result`, kết qủa thu được là các file text đã chứa nhận xét được gán nhãn.

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

## Customization
### Data Loader
* **Writing your own data loader**
  1. **Inherit ```BaseDataLoader```**

     ```BaseDataLoader``` handles:
     * Generating next batch
     * Data shuffling
     * Generating validation data loader ```BaseDataLoader.split_validation()```

  2. **Implementing abstract methods**

     There are some abstract methods you need to implement before using the methods in ```BaseDataLoader``` 
     * ```_pack_data()```: pack data members into a list of tuples
     * ```_unpack_data```: unpack packed data
     * ```_update_data```: updata data members
     * ```_n_samples```: total number of samples

* **DataLoader Usage**

  ```BaseDataLoader``` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to ```data_loader/data_loaders.py``` for an MNIST example

### Trainer
* **Writing your own trainer**
  1. **Inherit ```BaseTrainer```**

     ```BaseTrainer``` handles:
     * Training process logging
     * Checkpoint saving
     * Checkpoint resuming
     * Reconfigurable monitored value for saving current best 
       - Controlled by the arguments ```monitor``` and ```monitor_mode```, if ```monitor_mode == 'min'``` then the trainer will save a checkpoint ```model_best.pth.tar``` when ```monitor``` is a current minimum

  2. **Implementing abstract methods**

     You need to implement ```_train_epoch()``` for your training process, if you need validation then you can implement ```_valid_epoch()``` as in ```trainer/trainer.py```

* **Example**

  Please refer to ```trainer/trainer.py```

### Model
* **Writing your own model**
  1. **Inherit ```BaseModel```**

     ```BaseModel``` handles:
     * Inherited from ```torch.nn.Module```
     * ```summary()```: Model summary

  2. **Implementing abstract methods**

     Implement the foward pass method ```forward()```
     
* **Example**

  Please refer to ```model/model.py```

### Loss & Metrics
If you need to change the loss function or metrics, first ```import``` those function in ```train.py```, then modify:
```python
loss = my_loss
metrics = [my_metric]
```
They will appear in the logging during training
#### Multiple metrics
If you have multiple metrics for your project, just add them to the ```metrics``` list:
```python
loss = my_loss
metrics = [my_metric, my_metric2]
```
Additional metric will be shown in the logging
### Additional logging
If you have additional information to be logged, in ```_train_epoch()``` of your trainer class, merge them with ```log``` as shown below before returning:
```python
additional_log = {"gradient_norm": g, "sensitivity": s}
log = {**log, **additional_log}
return log
```
### Validation data
If you need to split validation data from a data loader, call ```BaseDataLoader.split_validation(validation_split)```, it will return a validation data loader, with the number of samples according to the specified ratio
**Note**: the ```split_validation()``` method will modify the original data loader
### Checkpoint naming
You can specify the name of the training session in ```train.py```
```python
training_name = type(model).__name__
```
Then the checkpoints will be saved in ```saved/training_name```

## Contributing
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

## TODOs
- [ ] Multi-GPU support
- [ ] `TensorboardX` support
- [ ] Support iteration-based training (instead of epoch)
- [ ] Load settings from `config` files
- [ ] Configurable logging layout
- [ ] Configurable checkpoint naming
- [ ] Options to save logs to file
- [ ] More clear trainer structure

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgments
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)