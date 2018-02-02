## 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。
wget http://thuctc.thunlp.org/source/THUCNews.zip

## 相关工具
sudo easy_install jieba           #python2

sudo pip install tflearn

sudo pip install gensim
wrong:no lapack/blas resources found

sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev
sudo pip install --upgrade pip

sudo pip install numpy
sudo pip install scipy

## 数据预处理

`copy_data.sh`    用于从每个分类拷贝一定数量的文件
`gene_train_data.py`    用于将多个文件整合到一个文件中。得到data.txt
`python word_clean_segment.py ../data/data.txt clean_seg_data.txt clean_seg_label.txt`
`python train_word2vec_model.py clean_seg_data.txt model.bin model.txt`

最终训练的数据集为:`clean_seg_data.txt`  
对应的标签:`clean_seg_label.txt`
训练的word2vec模型:` model.bin`
data字典:`my_vocab_512`  
label字典:`my_vocab_label`

训练：
python3 train.py

sequence_length  0.937
0.952
