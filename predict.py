#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

import sys
import os
import codecs
import tensorflow as tf
import numpy as np

from model import TextRNN, TestConfig
from data_util_zhihu import load_data_predict,load_final_test_data,create_voabulary,create_voabulary_label
from tflearn.data_utils import pad_sequences #to_categorical

tf.app.flags.DEFINE_string("ckpt_dir","text_rnn_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_string("traning_data_path","raw_data/word2vec/clean_seg_data_label.txt","path of traning data.") 
tf.app.flags.DEFINE_string("word2vec_model_path","raw_data/word2vec/model.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("predict_target_file","text_rnn_checkpoint/zhihu_result","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'test-zhihu-forpredict-v4only-title.txt',"target file path for final prediction")

def main(_):
     os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(0)
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_model_path=FLAGS.word2vec_model_path,name_scope="rnn")
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="rnn")
    testX, testY = load_data_predict(multi_label_flag, vocabulary_word2index,vocabulary_word2index_label)
    # 2.Data preprocessing: Sequence padding
    config = TestConfig()
    config.vocab_size = vocab_size
    testX2 = pad_sequences(testX, maxlen=config.sequence_length, value=0.)
    # 3.create session.
    sess_config=tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    with tf.Session(config=sess_config) as sess:
        textRNN=TextRNN(config)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint for TextRNN")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data=len(testX2);print("number_of_training_data:",number_of_training_data)
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        #for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
        for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
            logits=sess.run(textRNN.logits,feed_dict={textRNN.input_x:testX2[start:end],textRNN.dropout_keep_prob:1}) #'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            # 7. write question id and labels to file system.
            #############################################################################################################
            print("start:",start,";end:",end)
            question_id_sublist=question_id_list[start:end]
            get_label_using_logits_batch(question_id_sublist, logits, vocabulary_index2word_label, predict_target_file_f)
            ########################################################################################################
            index=index+1
        predict_target_file_f.close()

# get label using logits
def get_label_using_logits_batch(question_id_sublist,logits_batch,vocabulary_index2word_label,f,top_number=5):
    #print("get_label_using_logits.shape:", logits_batch.shape) # (10, 1999))=[batch_size,num_labels]===>需要(10,5)
    for i,logits in enumerate(logits_batch):
        index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
        index_list=index_list[::-1]
        label_list=[]
        for index in index_list:
            label=vocabulary_index2word_label[index]
            label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        #print("get_label_using_logits.label_list",label_list)
        write_question_id_with_labels(question_id_sublist[i], label_list, f)
    f.flush()
    #return label_list
# write question id and labels to file system.
def write_question_id_with_labels(question_id,labels_list,f):
    labels_string=",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

if __name__ == "__main__":
    tf.app.run()
