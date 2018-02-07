## 数据集

使用THUCNews的一个子集进行训练与测试，数据集下载：[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)

wget http://thuctc.thunlp.org/source/THUCNews.zip

## 相关工具安装
```shell
sudo easy_install jieba           #python2，中文分词工具
sudo pip install tflearn
sudo pip install gensim          #word2vec工具

sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev

sudo pip install --upgrade pip
sudo pip install numpy          
sudo pip install scipy
```

## 数据预处理

在raw_data里面是一些关于数据预处理的程序。

我们直接下载的THUCNews是一些中文新闻，新闻类别都已经分好了并且每个类别下有很多新闻。我们不需要这么多数据，所以只取其中的一个子集就可以了。

```shell
copy_data.sh              #用于从每个分类拷贝一定数量的文件
gene_train_data.py        #用于将多个文件整合到一个文件中。得到data.txt
```

经过上面的操作我们得到了训练文件data.txt，文件每一行的内容都为：类别名\t内容。其中内容是我们的feature，类别名是我们的target。

接下来我们对文本进行简单的清洗之后使用google的word2vec先训练出词向量。

```python
python word_clean_segment.py data.txt clean_seg_data.txt clean_seg_label.txt      #其中clean_seg_data.txt是清洗并且分词之后的文本，clean_seg_label.txt里面对应的是每一行文本的内容。

python train_word2vec_model.py clean_seg_data.txt model.bin model.txt
```

最终训练的数据集为:`clean_seg_data.txt`  

对应的标签:`clean_seg_label.txt`

训练的word2vec模型:` model.bin`

data字典:`my_vocab_512`  

label字典:`my_vocab_label`



#训练

`python3 train.py`

一些参数可以在train.py和config.py里面更改。

`python3 predict.py`

训练好模型之后可以使用这个程序去查看在不同数据集上的正确率。

最终正确率为：0.962
