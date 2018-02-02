#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import tensorflow as tf
import numpy as np

from model import TextRNN
from config import Config
from data_util import load_data_multilabel, create_voabulary, create_voabulary_label
from tflearn.data_utils import pad_sequences #to_categorical
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("ckpt_dir", "text_rnn_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_string("train_data_path", "raw_data/word2vec/clean_seg_data_label.txt", "path of traning data.") 
tf.app.flags.DEFINE_string("word2vec_model_path", "raw_data/word2vec/model.bin", "word2vec's vocabulary and vectors")

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(0)
    tensorboard_dir = 'tensorboard/text_rnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    #1.load data(X:list of lint,y:int).
    trainX, trainY, testX, testY = None, None, None, None
    vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_model_path=FLAGS.word2vec_model_path,name_scope="rnn")
    vocab_size = len(vocabulary_word2index)
    print("rnn_model.vocab_size:",vocab_size)
        
    vocabulary_word2index_label,vocabulary_index2word_label = create_voabulary_label(train_data_path=FLAGS.train_data_path, name_scope="rnn")
        
    train, test, _ =  load_data_multilabel(multi_label_flag=False, vocabulary_word2index=vocabulary_word2index, vocabulary_word2index_label=vocabulary_word2index_label, train_data_path=FLAGS.train_data_path) 
    trainX, trainY = train
    testX, testY = test
       
    config = Config()
    config.vocab_size = vocab_size 
    print("start padding & transform to one hot...")
    trainX = pad_sequences(trainX, maxlen=config.sequence_length, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=config.sequence_length, value=0.)  # padding to max length
    print("trainX[0]:", trainX[0]) 
    print("trainY[0]:", trainY[0])
    print("end padding & transform to one hot...")
    
    #2.create session.
    sess_config=tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    with tf.Session(config=sess_config) as sess:
        textRNN=TextRNN(config)
        tf.summary.scalar("cost", textRNN.loss_val)
        tf.summary.scalar("acc", textRNN.accuracy)
        tf.summary.scalar("lr", textRNN.learning_rate)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)

        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if config.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, config, vocabulary_index2word, vocab_size, textRNN,word2vec_model_path=FLAGS.word2vec_model_path)
 
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=config.batch_size
        lr = config.learning_rate
        best_loss = 0.0
        for epoch in range(0, config.max_epoch):
            loss, acc, counter = 0.0, 0.0, 0
            save_path=FLAGS.ckpt_dir+"model.ckpt"
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
                    print("trainY[start:end]:",trainY[start:end])
                curr_loss,curr_acc,_,summary=sess.run([textRNN.loss_val,textRNN.accuracy, textRNN.train_op, merged_summary],feed_dict={textRNN.input_x:trainX[start:end],textRNN.input_y:trainY[start:end]}) 
                writer.add_summary(summary, epoch*batch_size+start/batch_size)    #tensorboard
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                print("%d\t %.3f\t %.3f\t Train Loss: %.3f\t Train Accuracy:%.3f" %(epoch, lr, counter*batch_size/number_of_training_data, loss/float(counter), acc/float(counter))) 
            
            # 4.validation
            eval_loss, eval_acc=do_eval(sess,textRNN,testX,testY,batch_size,vocabulary_index2word_label)
            print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_acc))
            if epoch == 0 or ((best_loss - eval_loss) / eval_loss) < 0.05:
                saver.save(sess,save_path,global_step=epoch)
                best_loss = eval_loss
            else:
                lr = sess.run(tf.assign(textRNN.learning_rate, textRNN.learning_rate * config.decay_rate))

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, textRNN, testX, testY, batch_size, vocabulary_index2word_label)
    
def assign_pretrained_word_embedding(sess,config,vocabulary_index2word,vocab_size,textRNN,word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model = KeyedVectors.load(word2vec_model_path)
    word2vec_dict = {}
    for word in word2vec_model.wv.vocab.keys():
        word2vec_dict[word] = word2vec_model[word]
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(config.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, config.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
        #print(word_embedding_2dlist[i])
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    print(word_embedding_final.shape)
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度, eval的时候不计算train_op，也就是没有反向
def do_eval(sess,textRNN,evalX,evalY,batch_size,vocabulary_index2word_label):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        curr_eval_loss, logits,curr_eval_acc= sess.run([textRNN.loss_val,textRNN.logits,textRNN.accuracy],
                                          feed_dict={textRNN.input_x: evalX[start:end],textRNN.input_y: evalY[start:end]})
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

if __name__ == "__main__":
    tf.app.run()
