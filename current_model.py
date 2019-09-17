# coding: utf-8
import tensorflow as tf
import numpy as np
import time
from utils import ps_pb_interaction, pq_interaction, source2token, dense

class Config(object):
    word2vec_init = True
    # word2vec_size = 300
    # hidden_size = 300
    word2vec_size = 50
    hidden_size = 50
    batch_size = 8
    learning_rate = 0.05
    l2_weight = 1e-6
    dropout = 1.0
    max_sent = 150
    max_word = 20
    label_list = ['Good', 'PotentiallyUseful', 'Bad']
    write_list = ['true', 'false', 'false']
    restore = False
    gpu_id = 0

    char_emb = 100
    filter_sizes = [2, 3, 4, 5]
    filter_num = 25


class Model(object):
    # def __init__(self, embedding):
    def __init__(self, embedding):
        self.word_embedding = embedding

    def build_model(self):  # 数据从generator带过来，然后喂到字典中
        with tf.variable_scope("attention_model", initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.ps_words = tf.placeholder(tf.int32, [None, None])                       # (b,m)
            self.pb_words = tf.placeholder(tf.int32, [None, None])                       
            self.qt_words = tf.placeholder(tf.int32, [None, None])                    

            self.ps_length= tf.reduce_sum(tf.sign(self.ps_words), 1)                       # (b,)
            self.pb_length= tf.reduce_sum(tf.sign(self.pb_words), 1)
            self.qt_length= tf.reduce_sum(tf.sign(self.qt_words), 1)                       # (b,m,1)
            self.ps_mask = tf.expand_dims(tf.sequence_mask(self.ps_length, tf.shape(self.ps_words)[1], tf.float32), -1)  # 最后一维的数据分裂独立出来
            self.pb_mask = tf.expand_dims(tf.sequence_mask(self.pb_length, tf.shape(self.pb_words)[1], tf.float32), -1)
            self.qt_mask = tf.expand_dims(tf.sequence_mask(self.qt_length, tf.shape(self.qt_words)[1], tf.float32), -1)

            self.is_train = tf.placeholder(tf.bool)
            self.dropout = tf.cond(self.is_train, lambda: Config.dropout, lambda: 1.0)
            self.labels = tf.placeholder(tf.int32, [None])                          # (b,)

            with tf.device('/cpu:0'):  # 单词级别的嵌入
                # self.embed_matrix = tf.convert_to_tensor(self.word_embedding, dtype=tf.float32)
                self.embed_matrix = self.word_embedding
                self.ps_emb = tf.nn.embedding_lookup(self.embed_matrix, self.ps_words)        # (b,m,d)  subject词向量
                self.pb_emb = tf.nn.embedding_lookup(self.embed_matrix, self.pb_words)        # body嵌入
                self.qt_emb = tf.nn.embedding_lookup(self.embed_matrix, self.qt_words)        # answer嵌入

            with tf.variable_scope("pq_interaction"):
                para = ps_pb_interaction(self.ps_emb, self.pb_emb, self.ps_mask, self.pb_mask, self.dropout, 'parallel')  # Spara
                orth = ps_pb_interaction(self.ps_emb, self.pb_emb, self.ps_mask, self.pb_mask, self.dropout, 'orthogonal')  # Sorth
                self.p = tf.concat([para, orth], -1)  # Srep

                # 输入输出激活函数
                self.q = tf.layers.dense(self.qt_emb, 2*Config.hidden_size, tf.nn.tanh, name='qt_tanh') * tf.layers.dense(self.qt_emb,2*Config.hidden_size,tf.nn.sigmoid, name='qt_sigmoid')
                p_inter, q_inter = pq_interaction(self.p, self.q, self.ps_mask, self.qt_mask, self.dropout, 'p_q')  # 输入Srep，Crep，输出Satt和Catt
                self.p_vec = source2token(p_inter, self.ps_mask, self.dropout, 'p_vec')  # 返回Ssum
                self.q_vec = source2token(q_inter, self.qt_mask, self.dropout, 'q_vec')  # 返回Csum

            with tf.variable_scope("loss"):
                l0 = tf.concat([self.p_vec, self.q_vec], 1)  # Ssum and Csum
                l1 = tf.layers.dense(l0, 300, tf.nn.elu, name='l1')  # 第一层
                l2 = tf.layers.dense(l1, 300, tf.nn.elu, name='l2')  # 第二层
                self.logits = tf.layers.dense(l2, 3, tf.identity, name='logits')  # 第三层输出
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, 3, dtype=tf.float32), logits=self.logits), -1)
                                                                                                    # 深度为3，默认axis=-1（最深的维度）

                for v in tf.trainable_variables():
                    self.loss += Config.l2_weight * tf.nn.l2_loss(v)  # 添加正则损失， 所有变量的L2范数
                self.train_op = tf.train.GradientDescentOptimizer(Config.learning_rate).minimize(self.loss)

    def train_batch(self, sess, batch_data):
        ps_words, pb_words, qt_words, labels, dialogue_ids = batch_data
        feed = {self.ps_words: ps_words,
                self.pb_words: pb_words,
                self.qt_words: qt_words,
                self.labels: labels,
                self.is_train: True
               }
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss,

    def test_batch(self, sess, batch_test, is_deva=False):
        ps_words, pb_words, qt_words, label, cids = batch_test
        feed = {self.ps_words: ps_words,
                self.pb_words: pb_words,
                self.qt_words: qt_words,
                # self.ps_chars: ps_chars,
                # self.pb_chars: pb_chars,
                # self.qt_chars: qt_chars,
                self.is_train: False
               }
        logits = sess.run(self.logits, feed_dict = feed)  # 8行3列 样本数*分类数目
        score = logits[:, 0]  # 取出第一列
        predict = np.argmax(logits, 1)  # 返回每一行 最大值所在位置的索引 列表--->最大的地方就是预测的位置
        return_string = ''
        for i in range(len(cids)):
            if is_deva:
                wstring = cids[i]+'\t'+ Config.label_list[predict[i]]+'\n'
            else:
                wstring = '_'.join(cids[i].split('_')[:2])+'\t'+cids[i]+'\t0\t'+str(score[i])+'\t'+Config.write_list[predict[i]]+'\n'
            return_string += wstring
        return return_string
