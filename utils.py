# original paper setting: -/77.21/87.85
# add char-cnn 200d: 57.30/77.64/87.75
# 1 layer highway body->subject<->comment: 56/76/86
# 1 layer unshared tanh body->subject<->comment: 57.51/77.18/86.41
import tensorflow as tf


# inpa,inpb (b,m,d) maska,maskb (b,m,1)
def ps_pb_interaction(ps, pb, ps_mask, pb_mask, keep_prob, scope):
    """
    Question Condensing 部分
    :param ps:
    :param pb:
    :param ps_mask:
    :param pb_mask:
    :param keep_prob:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        b, m, n, d = tf.shape(ps)[0], tf.shape(ps)[1], tf.shape(pb)[1], ps.get_shape().as_list()[2]
        attn_mask = tf.expand_dims(ps_mask * tf.reshape(pb_mask, [b, 1, n]), -1)     # (b,m,n,1)   (b,1,n --->b,m,n--->b,m,n,1)

        head = tf.tile(tf.expand_dims(ps, 2), [1, 1, n, 1])                          # (b,m,1,d)
        tail = tf.tile(tf.expand_dims(pb, 1), [1, m, 1, 1])                          # (b,1,n,d)
        parallel = head * tf.reduce_sum(head * tail, -1, True)/(tf.reduce_sum(head * head, -1, True)+1e-5)  # 公式1
        orthogonal = tail - parallel                                                                        # 公式2
        base = parallel if scope == 'parallel' else orthogonal

        interaction = dense(base, d, scope='interaction')    # (b,m,n,d)
        logits = 10.0*tf.tanh((interaction)/10.0) + (1 - attn_mask) * (-1e30)                               # 公式3

        attn_score = tf.nn.softmax(logits, 2) * attn_mask                                                   # 公式4
        attn_result = tf.reduce_sum(attn_score * tail, 2)                           # (b,m,d)               # 公式5
        fusion_gate = dense(tf.concat([ps, attn_result], -1), d, tf.sigmoid, scope='fusion_gate')*ps_mask   # 公式6
        return (fusion_gate*ps + (1-fusion_gate)*attn_result) * ps_mask                                     # 公式7

# inpa,inpb (b,m,d) maska,maskb (b,m,1)
def pq_interaction(ps, qt, ps_mask, qt_mask, keep_prob, scope):
    """
    问答交互部分，返回Satt和Catt
    :param ps:         Srep
    :param qt:         Crep
    :param ps_mask:
    :param qt_mask:
    :param keep_prob:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        b, m, n, d = tf.shape(ps)[0], tf.shape(ps)[1], tf.shape(qt)[1], ps.get_shape().as_list()[2]
        attn_mask = tf.expand_dims(ps_mask*tf.reshape(qt_mask,[b,1,n]), -1)             # (b,m,n,1)

        head = tf.tile(tf.expand_dims(ps, 2), [1,1,n,1])                          # (b,m,1,d)
        tail = tf.tile(tf.expand_dims(qt, 1), [1,m,1,1])                          # (b,1,n,d)

        interaction = dense(tf.concat([head, tail], -1), d, scope='interaction')    # (b,m,n,d)         公式9
        #interaction = tf.reduce_sum(head*tail, -1, True)
        #interaction = tf.reduce_sum(dense(head, d, scope='interaction')*tail, -1, True)

        logits = 5.0*tf.tanh((interaction)/5.0) + (1 - attn_mask) * (-1e30)                           # 公式10

        atta_score = tf.nn.softmax(logits, axis=2) * attn_mask                                             # 公式11左部权重计算
        atta_result = tf.reduce_sum(atta_score * tail, axis=2)                           # (b,m,d)         # 公式11--->S ai

        attb_score = tf.nn.softmax(logits, 1) * attn_mask                                             # 公式12左部权重计算
        attb_result = tf.reduce_sum(attb_score * head, 1)                       # (b,n,d)               公式12--->C ai

        cata = tf.concat([ps, atta_result], -1) * ps_mask   # S att
        catb = tf.concat([qt, attb_result], -1) * qt_mask   # C att
        return cata, catb  # Satt和Catt


def source2token(rep_tensor, rep_mask, keep_prob, scope):
    """

    :param rep_tensor: Satt
    :param rep_mask:
    :param keep_prob:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        ivec = rep_tensor.get_shape().as_list()[2]
        map1 = dense(rep_tensor, ivec, tf.nn.elu, keep_prob, 'map1')*rep_mask  # (b,n,d)  # 公式13
        map2 = dense(rep_tensor, ivec, tf.identity, keep_prob, 'map2')*rep_mask  # (b,n,d)
        map2_masked = map1 + (1-rep_mask) * (-1e30)
        soft = tf.nn.softmax(map2_masked, 1)*rep_mask   # bs,sl,vec
        return tf.reduce_sum(soft * rep_tensor, 1)      # bs, vec  # 公式14


def dense(inp, out_size, activation=tf.identity, keep_prob=1.0, scope=None, need_bias=True):
    with tf.variable_scope(scope):
        inp_shape = [inp.get_shape().as_list()[i] or tf.shape(inp)[i] for i in range(len(inp.get_shape().as_list()))]  # 获取shape列表
        input = tf.nn.dropout(tf.reshape(inp, [-1, inp_shape[-1]]), keep_prob)
        W = tf.get_variable('W', shape=[input.get_shape()[-1],out_size],dtype=tf.float32)
        b = tf.get_variable('b', shape=[out_size], dtype=tf.float32, initializer=tf.zeros_initializer()) if need_bias else 0
        return activation(tf.reshape(tf.matmul(input, W) + b, inp_shape[:-1] + [out_size]))



    

