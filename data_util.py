# -*- coding: utf-8 -*-

import codecs
import numpy as np
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import os
import pickle
from tflearn.data_utils import pad_sequences

_PAD="_PAD"
PAD_ID = 0

def create_voabulary(word2vec_model_path, name_scope=''):
    vocabulary_word2index={}
    vocabulary_index2word={}
    print("create vocabulary. word2vec_model_path:",word2vec_model_path)
    model = KeyedVectors.load(word2vec_model_path)
    vocabulary_word2index['PAD_ID']=0
    vocabulary_index2word[0]='PAD_ID'
    special_index=0
    if 'biLstmTextRelation' in name_scope:
        vocabulary_word2index['EOS']=1 # a special token for biLstTextRelation model. which is used between two sentences.
        vocabulary_index2word[1]='EOS'
        special_index=1
    for i,vocab in enumerate(model.wv.vocab.keys()):
        vocabulary_word2index[vocab]=i+1+special_index
        vocabulary_index2word[i+1+special_index]=vocab

    return vocabulary_word2index,vocabulary_index2word

# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is low frequency.
def create_voabulary_label(train_data_path, name_scope=''):
    print("create_voabulary_label_sorted.started.train_data_path: ", train_data_path)
    train_label = codecs.open(train_data_path, 'r', 'utf8')
    lines = train_label.readlines()
    count = 0
    vocabulary_word2index_label = {}
    vocabulary_index2word_label = {}
    vocabulary_label_count_dict = {} #{label:count}
    for i, line in enumerate(lines):
        line = line.strip().replace("\n","")
        label = line.split('\t')
        if vocabulary_label_count_dict.get(label[1],None) is not None:
            vocabulary_label_count_dict[label[1]]=vocabulary_label_count_dict[label[1]]+1
        else:
            vocabulary_label_count_dict[label[1]]=1
    list_label=sort_by_value(vocabulary_label_count_dict)

    print("length of list_label:",len(list_label));
    countt=0

    for i,label in enumerate(list_label):
        if i<10:
            count_value=vocabulary_label_count_dict[label]
            print("label:",label,"count_value:",count_value)
            countt=countt+count_value
        indexx = i
        vocabulary_word2index_label[label]=indexx
        vocabulary_index2word_label[indexx]=label
    print("count top10:",countt)

    print("create_voabulary_label_sorted.ended.len of vocabulary_label:",len(vocabulary_index2word_label))
    return vocabulary_word2index_label,vocabulary_index2word_label

def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [ backitems[i][1] for i in range(0,len(backitems))]

def load_data_multilabel(multi_label_flag, vocabulary_word2index, vocabulary_word2index_label, train_data_path): 
    """
    input: a file path
    return: train, test, valid. train=(trainX, trainY).
    trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load data from file
    print("load_data.started...")
    data_f = codecs.open(train_data_path, 'r', 'utf8') 
    lines = data_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    for i, line in enumerate(lines):
        x, y = line.split('\t') 
        y = y.strip().replace('\n','')
        x = x.strip()
        if i < 1:
            print(i,"x0:",x) 
        x = x.split(" ")
        x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        if i < 2:
            print(i,"x1:",x) 
        if multi_label_flag: # 2)prepare multi-label format for classification
            ys = y.replace('\n', '').split(" ")  # ys is a list
            ys_index=[]
            for y in ys:
                y_index = vocabulary_word2index_label[y]
                ys_index.append(y_index)
            ys_mulithot_list=transform_multilabel_as_multihot(ys_index)
        else:                #3)prepare single label format for classification
            ys_mulithot_list=vocabulary_word2index_label[y]
        if i < 3:
            print(i,"y:",y," ;ys_mulithot_list:",ys_mulithot_list) #," ;ys_decoder_input:",ys_decoder_input)
        X.append(x)
        Y.append(ys_mulithot_list)
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:",number_examples)
    valid_portion = 0.05 
    train = (X[0:int((1 - valid_portion) * number_examples)], Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])
    # 5.return
    print("load_data.ended...")
    return train, test, test

def load_data_predict(multi_label_flag, vocabulary_word2index,vocabulary_word2index_label):  
    data_f = codecs.open(train_data_path, 'r', 'utf8') 
    lines = data_f.readlines()
    X = []
    Y = []
    for i, line in enumerate(lines):
        x, y = line.split('\t') 
        y = y.strip().replace('\n','')
        x = x.strip()
        x = x.split(" ")
        x = [vocabulary_word2index.get(e,0) for e in x]
        if multi_label_flag: 
            ys = y.replace('\n', '').split(" ")  
            ys_index=[]
            for y in ys:
                y_index = vocabulary_word2index_label[y]
                ys_index.append(y_index)
            ys_mulithot_list=transform_multilabel_as_multihot(ys_index)
        else:              
            ys_mulithot_list=vocabulary_word2index_label[y]
        X.append(x)
        Y.append(ys_mulithot_list)
    number_examples = len(X)
    print("number_examples:",number_examples)
    return X, Y

#将LABEL转化为MULTI-HOT
def transform_multilabel_as_multihot(label_list,label_size=1999): #1999label_list=[0,1,4,9,5]
    """
    :param label_list: e.g.[0,1,4]
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

def proces_label_to_algin(ys_list,require_size=5):
    """
    :param ys_list: a list
    :return: a list
    """
    ys_list_result=[0 for x in range(require_size)]
    if len(ys_list)>=require_size: #超长
        ys_list_result=ys_list[0:require_size]
    else:#太短
       if len(ys_list)==1:
           ys_list_result =[ys_list[0] for x in range(require_size)]
       elif len(ys_list)==2:
           ys_list_result = [ys_list[0],ys_list[0],ys_list[0],ys_list[1],ys_list[1]]
       elif len(ys_list) == 3:
           ys_list_result = [ys_list[0], ys_list[0], ys_list[1], ys_list[1], ys_list[2]]
       elif len(ys_list) == 4:
           ys_list_result = [ys_list[0], ys_list[0], ys_list[1], ys_list[2], ys_list[3]]
    return ys_list_result




