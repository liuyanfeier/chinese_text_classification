# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class TextRNN:
    def __init__(self, config):
        # set hyperparamter
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.max_grad_norm = config.max_grad_norm
        self.num_layers = config.num_layers
        self.sequence_length = config.sequence_length
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.is_training = config.is_training
        self.initializer = tf.random_normal_initializer(stddev=config.init_scale)
        
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False, dtype=tf.float32)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y")  # y [None,num_classes]

        self.instantiate_weights()
        self.logits = self.inference(config) #[None, self.label_size].
        if not config.is_training:
            return
        
        self.loss_val = self.loss()
        #self.train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=0.01, optimizer="Adam")
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_val, tvars), self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
        
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
    
    def instantiate_weights(self):
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] 
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size]

    def inference(self, config):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) #shape:[None,sentence_length,embed_size]
        if config.dropout_keep_prob < 1 and config.is_training:
            self.embedded_words = tf.nn.dropout(self.embedded_words, config.keep_prob)        

        def make_cell(): 
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_size) 
            if config.dropout_keep_prob < 1 and config.is_training:
                lstm_cell=tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob=config.dropout_keep_prob)
            return lstm_cell
       
        lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        
        # get the length of each sample
        #self.length = tf.reduce_sum(tf.sign(self.input_x), axis=1)
        #self.length = tf.cast(self.length, tf.int32) #类型转换 
        #print("self.length:===>", self.length)

        #init_state_fw = lstm_fw_cell.zero_state(self.batch_size, tf.float32) 
        #init_state_bw = lstm_bw_cell.zero_state(self.batch_size, tf.float32) 
        #outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,sequence_length=self.length,
        #                                          initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, dtype=tf.float32)
        outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype=tf.float32)
        print("outputs:===>", outputs)
        
        output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
        self.output_rnn_last=tf.reduce_mean(output_rnn,axis=1) 
        print("output_rnn_last:", self.output_rnn_last)
 
        with tf.name_scope("output"): #inputs: A Tensor of shape [batch_size, dim]
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self,l2_lambda=0.0000):
        with tf.name_scope("loss"):
            #input: logits and labels must have the same shape [batch_size, num_classes]
            #output: A 1-D Tensor of length batch_size of the same type as logits with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            for v in tf.trainable_variables():
                print(v)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss


