import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/",one_hot = True)

#设置训练参数
learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 10

#设置输入的列，行，隐藏层单元输，分类数
n_input = 28
n_steps=28
n_hidden = 256
n_classes = 10

#输入
x = tf.palceholder("float",[None,n_steps,n_input])  #二维结构 None个（长，宽）
y = tf.placeholder("float",[None,n_classes])

#sofrmax weight bias
weights = tf.Variable(tf.random_normal([2*n_hideen,n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

def