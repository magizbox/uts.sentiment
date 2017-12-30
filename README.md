# Sentiment Analysis Experiments

This repository contains experiments in Vietnamese sentiment analysis problems. It is a part of [underthesea](https://github.com/magizbox/underthesea) project.

# Results

![](https://img.shields.io/badge/F1-0.43-red.svg)

| Model                                                          | F1 Score |
|----------------------------------------------------------------|----------|
| Logistic Regression (Tfidf_ngrams(1,2) + max_df=0.8+ min_df=8) | 0.53     |
| Logistic Regression (Count_ngrams(1,2) + max_df=0.8+ min_df=8) | 0.58     |
| SVM (Count_ngrams(1,2) + max_df=0.8 + min_df=0.005)            | 0.58     |
| SVM (Count_ngrams(1,2) + max_df=0.5 + min_df=0.005)            | 0.59     |


# Reproduce

Setup Environment

```
# clone project
$ git clone https://github.com/undertheseanlp/sentiment

# create environment
$ cd sentiment
$ conda create -n sentiment python=3.5
$ pip install -r requirements.txt
$ pip install git+https://github.com/undertheseanlp/languageflow
```

# Related Works

* Vietnamese Sentiment Analysis publications, [link](https://github.com/magizbox/underthesea/wiki/Vietnamese-NLP-Publications#sentiment-analysis)
