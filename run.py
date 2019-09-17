#coding: utf-8
# v1.2 也可以先只取出每个句子，在batch_iter阶段转为id
# 这样好处是不需要一开始就将整个数据集进行转换，减小等待时间，减少内存消耗；
# 坏处是对每个句子每个epoch都要转换一遍，增加了整体的运行时间；

import tensorflow as tf
import numpy as np
import os 
import sys
import time
import matplotlib.pyplot as plt
from current_model import Config
from current_model import Model

# os.environ['CUDA_VISIBLE_DEVICES'] = str(3-Config.gpu_id)  # by RBX
vocab = {}

def plot_matrix(losses):
    plt.plot(losses)
    plt.title('lost_curve')
    plt.ylabel('batch_loss')
    plt.xlabel('step')
    plt.show()


def get_word_and_char(sentence):
    """
    语句编码函数 word--->索引
    :param sentence: 一句话--->空格分割的字符串
    :return:一句话对应的词向量索引列表 和字符二维列表（内层不等长）
    """

    words = sentence.lower().split()                          # 单词列表
    get_id = lambda w: vocab[w] if w in vocab else 1  # 如果当前单字在字典中返回其索引，否则返回默认索引1，字典外的单字都为1
    word_ids = [get_id(w) for w in words]             # 返回这句话单词的索引列表，如果不在字典中，就设置索引为1.
    # digitize_char = lambda x: ord(x) if 0 < ord(x) < 128 else 1  # 如果字符在ascall中返回其索引，不在的话返回1
    # char_ids = [[digitize_char(c) for c in w] for w in words]    # 返回二维列表，内层列表是每个单词拆分成字符后的ascall索引
    # return word_ids, char_ids
    return word_ids


def digitize_data(fname):
    """
    文件解析函数

    把当前文件内的每一行 解析成一个列表：

                    [主题索引列表，主题字符列表,
                     body索引列表，body字符列表,
                     answer索引列表，answer字符列表,
                     label列表   ]

    然后把列表嵌套后返回
    """
    file_parsed = []
    start_time = time.time()
    with open('data/xml/' + fname + '.txt') as fi:
    # with open('data/' + fname + '.txt') as fi:
        for line in fi:
            a = line.strip().split('\t')
            dialogue_id, label, subject_string, body_string, answer_string = a
            subject_word_list = get_word_and_char(subject_string)  # Subject 字符串对应的索引列表 以及 字符二维列表
            body_word_list = get_word_and_char(body_string)           # Body    字符串对应的索引列表 以及 字符二维列表
            answer_word_list = get_word_and_char(answer_string)     # Answer  字符串对应的索引列表 以及 字符二维列表
            label = Config.label_list.index(label) if fname == '15train' else label
            file_parsed.append([subject_word_list, body_word_list, answer_word_list, label, dialogue_id])  # 注最后的dialogue_id由rbx添加

    print('解析文件 {}......:\n长度是 :{}\t耗时:{}'.format(fname, len(file_parsed), time.time()-start_time))
    return file_parsed


# padding成固定shape,(b,m) (b,m,w),虽然各batch的b,m,w可能不同,
def batch_iter(data, batch_size, shuffle, is_train, epochs):
    """
    批量数据 generator
    :param data:  # 三维列表，二层针对每一行，最底层是一段材料的各个section-->subject、
                    body、answer对应的词向量list、字符索引二维lists，以及label，dialogue_id
    :return: padding后的各个矩阵
    """
    batchs = int((len(data)-1)/batch_size) + 1  # 批次数
    num_epochs = epochs if is_train else 1         # 设定圈数
    for epoch in range(num_epochs):
        # if shuffle:
        #     random.shuffle(data)
        for batch_num in range(batchs):                           # 计算批次号
            start_index = batch_num * batch_size                  # 定义切片起点
            end_index = min((batch_num+1)*batch_size, len(data))  # 定义切片终点
            batch_data = data[start_index:end_index]              # 数据集切片

            zip_data = zip(*batch_data)                           # *操作相当于解压->列表降维，面向各行做zip，zip后把相同内涵的列表重构成元组
            data_list = list(zip_data)
            # 元组构成的列表，元组内涵不同分别是 主题word索引列表、字符列表|body索引列表、字符列表|answer索引列表、字符列表|label
            # 数据结构： 列表（例如词向量索引）-->元组（因为批次数据对应多行）-->列表（因为数据有多个内涵）

            # subject_words, subject_chars, body_words, body_chars, answer_words, answer_chars, label, dialogue_id = data_list  # 批次数据降维-->上行注释的元组层
            subject_words, body_words, answer_words, label, dialogue_id = data_list  # 批次数据降维-->上行注释的元组层

            subject_words_matrix = pad_word_and_char(subject_words)  # 批次subject词向量二维矩阵、 词向量字符三维矩阵
            body_words_matrix = pad_word_and_char(body_words)        # 批次body   词向量二维矩阵、 词向量字符三维矩阵
            answer_words_matrix = pad_word_and_char(answer_words)      # 批次answer 词向量二维矩阵、 词向量字符三维矩阵
            yield [subject_words_matrix, body_words_matrix, answer_words_matrix, np.array(label), np.array(dialogue_id)]


def pad_word_and_char(words):
    """
    fun：数据padding

    :param words: 元组，内层为列表 <--- 批次中不同的行 某一字段（subject、body、answer） 对应的索引
    :param chars: 元组，内层为三维列表 <--- 批次中不同的行 某一字段（subject、body、answer） 对应的字符索引二维列表

    :return:填充后的句子和单词， 以最大长度为基准，短的补0
        np.array(padded_sent) shape = 批次容量 字段长度（字段单词个数）
        np.array(padded_word) shape = 批次容量 字段长度 单词长度
        返回批次词向量二维矩阵、 词向量字符三维矩阵
    """
    max_sentence = min(len(max(words, key=len)), Config.max_sent)  # 计算最大的字段长度
    # max 函数返回words中最长的列表（即表示最长的句子），求其长度与设定的上限比较 去两者的小值

    pad_sentecce = lambda x: x[:max_sentence] if len(x) > max_sentence else x+[0]*(max_sentence-len(x))  # 句子padding函数
    padded_sentence = [pad_sentecce(sentence) for sentence in words]  # 对字段索引列表padding， 短的补0  sent是列表层
    return np.array(padded_sentence)


if __name__ == '__main__':
    start_time = time.time()
    embedding = np.zeros((400005, 50), dtype=np.float32)
    # embedding = [[0]*50]
    # embedding = np.zeros((90, 300), dtype=np.float32)

    # with open('data/glove.840B.300d.txt', encoding='utf-8') as fe:
    # with open('data/123.txt', encoding='utf-8') as fe:
    with open('data/embedding_50d.txt', encoding='utf-8') as fe:
        """
        :embedding: 词向量列表的列表
        :vocab: 所有词构成的字典（word： index），索引和embedding列表中的位置对应
        """

        for i, line in enumerate(fe):  # 返回枚举类型(int索引 和 fe内容 构成)
            try:
                items = line.strip().split()  # 返回分割后的 列表
                word_embeding = list(map(float, items[1:]))
                if len(word_embeding) == 50:
                    embedding[i + 1] = word_embeding  # 单词的嵌入向量
                    # embedding.append(word_embeding)  # 单词的嵌入向量
                    vocab[items[0]] = i + 1  # 推测 :--->vocab : 单词和 索引字典
            except:
                pass
    print('加载词向量文件......\n长度:{}\t耗时:{}'.format(len(embedding), time.time() - start_time))
    model = Model(embedding)

    train_data = digitize_data('15train')  # 三维列表，二层针对每一行，最底层是一段材料的各个section-->subject、body、answer对应的词向量list、字符索引二维lists，以及label
    print(vars(Config))

    tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True  rbx

    with tf.Session(config=tf_config) as sess:
        model.build_model()              # 搭建模型
        for v in tf.trainable_variables():
            print('name:{}\tshape:{}'.format(v.name, v.shape))
        print("ps_gating_pb p_concat_q")

        sess.run(tf.global_variables_initializer())  # 初始化变量

        saver = tf.train.Saver()                     # 模型加载
        if Config.restore and len([v for v in os.listdir('weights/') if '.index' in v]):
            saver.restore(sess, tf.train.latest_checkpoint('weights/'))

            # 验证集
            deva_data = digitize_data('15dev')
            batch_devas = batch_iter(deva_data, Config.batch_size, False, False, epochs=1)
            with open('scorer/15dev/predict', 'w') as fw:
                for batch_deva in batch_devas:
                    fw.write(model.test_batch(sess, batch_deva, True))
        else:
            # 批量数据generator， padding后的各个矩阵构成的列表
            batch_trains = batch_iter(train_data, Config.batch_size, True, True, epochs=500)  # 参数：训练数据 批量 shuffle is_train
            loss = []
            losses = []
            show_time = time.time()
            total_steps = len(train_data)/Config.batch_size  # 批次数

            best_deva = best_devb = steps = steps2 = 0

            """模型训练"""
            for step, batch_train in enumerate(batch_trains):
                steps += 1
                steps2 += 1
                # batch_train 连接generator， 表示一批训练数据，不是yield中的一个矩阵
                batch_loss = model.train_batch(sess, batch_train)  # 数据构成：主题、body、answer3二维矩阵+3字符三维矩阵+ label
                sys.stdout.write("\repoch:{:.4f}\t\t\tloss:{:.4f}".format(step/total_steps, batch_loss[0]))
                loss.append(batch_loss[0])

                if steps + 1 > total_steps:
                    avg = np.mean(loss)
                    losses.append(avg)
                    loss = []
                    steps = 0

                if steps2 + 1 > 50 * total_steps:
                    saver.save(sess, 'weights/model.ckpt',step)
                    steps2 = 0
                    plot_matrix(losses)

            print('\n总耗时:{}'.format(time.time() - show_time))

