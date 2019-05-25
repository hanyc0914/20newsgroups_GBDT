# 20newspaper_GBDT

This is a implemetation of GBDT multi-classification with sklearn，the dataset is [20news-19997.tar.gz](http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz) 

## Usage
### Requirements
- python 3.6
  - nltk
  - scikit-learn
  - numpy
  - pickle
  
## Run

### 1. Generate train data & test data
need download and unzip 20news-19997.tar.gz to root dir(eg. './20_newsgroups'), and download crawl-300d-2M.vec, put it in './vectors/'.
```
python data_process.py
```
you will get 2 dirs, each with 4 pickle file：
- ./data_tfidf: word used TD-IDF 
- ./data: word embedding using fasttext, pretrained file: [crawl-300d-2M.vec](https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m)

### 2. Start training GBDT
```
python main.py
```

## Result
### 1. GBDT + fasttext:
![image](https://github.com/hanyc0914/20newsgroups_GBDT/blob/master/img/default_fasttext.png)

### 2. GBDT + TF-IDF:
![image](https://github.com/hanyc0914/20newsgroups_GBDT/blob/master/img/default_tfidf.png)

### 3. GBDT(n_estimators: 60, max_depth: 6) + TF-IDF
![image](https://github.com/hanyc0914/20newsgroups_GBDT/blob/master/img/default_tfidf_new.png)
