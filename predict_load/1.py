# coding = utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import random

f = open('data/ID01.csv')
df = pd.read_csv(f)
#df = df[df['ID'] == 1]
data = np.array(df['workload'])

normalize_data = (data - np.mean(data)) / np.std(data)
print(normalize_data)
normalize_data = normalize_data[:, np.newaxis]
print(normalize_data)

normalize_data2 = normalize_data[2000:]
normalize_data = normalize_data[:2000]

time_step = 80
rnn_unit = 128
batch_size = 20
input_size = 1
output_size = 1
lr = 0.0006
train_x, train_y = [], []
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
train_data = [[a, b] for a, b in zip(train_x, train_y)]
print(train_data[:10])
# 定义神经网络变量
X = tf.placeholder(tf.float32, [None, time_step, input_size])
Y = tf.placeholder(tf.float32, [None, time_step, output_size])
# 定义输入输出层的权重、偏置
# 输入输出到底是什么样的？？？
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


def lstm(batch):
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(200):  # We can increase the number of iterations to gain better result.
            random.shuffle(train_data)
            train_x = [a[0] for a in train_data]
            train_y = [a[1] for a in train_data]
            print(train_x[:10])
            print(train_y[:10])
            step = 0
            start = 0
            end = start + batch_size
            while (end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size

                if step % 10 == 0:
                    print("Number of iterations:", i, " loss:", loss_)
                if step % 100 == 0:
                    print("model_save", saver.save(sess, './model_save1/modle.ckpt'))
                    # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
                    # if you run it in Linux,please use  'model_save1/modle.ckpt'
                step += 1
        print("The train has finished")


train_lstm()

#在预测时计算每次预测的误差，最后得出均方根误差MSE的值
def prediction():
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(1)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, './model_save1/modle.ckpt')
        # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        # if you run it in Linux,please use  'model_save1/modle.ckpt'
        prev_seq = train_x[-1]
        predict = []

        count = 0
        SMAPE = 0
        for i in range(len(normalize_data2)):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            #RE为相对误差
            # print(next_seq[-1], normalize_data2[i], normalize_data2[i+1])
            SMAPE += math.fabs(next_seq[-1] - normalize_data2[i]) / (math.fabs(next_seq[-1]) + math.fabs(normalize_data2[i]))
            RE = math.fabs(next_seq[-1] - normalize_data2[i])/normalize_data2[i]
            #如果相对误差小于5%，那么计算结果正确，否则，错误
            if RE < 0.05:
                count += 1
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        print("The number of accuracy is :", count)
        accuracy = count/len(normalize_data2)
        print(SMAPE/len(normalize_data2))
        print("The accuracy of this model is :", accuracy)
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), normalize_data2[:len(normalize_data2)], color='b')
        plt.show()

        prev_seq = train_x[500]
        predict = []

        count = 0
        SMAPE = 0
        for i in range(len(normalize_data)):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            #RE为相对误差
            # print(next_seq[-1], normalize_data2[i], normalize_data2[i+1])
            SMAPE += math.fabs(next_seq[-1] - normalize_data[i]) / (math.fabs(next_seq[-1]) + math.fabs(normalize_data[i]))
            RE = math.fabs(next_seq[-1] - normalize_data[i])/normalize_data[i]
            #如果相对误差小于5%，那么计算结果正确，否则，错误
            if RE < 0.05:
                count += 1
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        print("The number of accuracy is :", count)
        accuracy = count/len(normalize_data)
        print(SMAPE/len(normalize_data))
        print("The accuracy of this model is :", accuracy)
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data))), predict, color='r')
        # plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), normalize_data2[:len(normalize_data2)], color='b')
        plt.show()


prediction()
