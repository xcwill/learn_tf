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
x = tf.placeholder("float",[None,n_steps,n_input])  #二维结构 batch_size个（长，宽）
y = tf.placeholder("float",[None,n_classes])

#sofrmax weight bias
weights = tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def BiRNN(x,weights,biases):

    x = tf.transpose(x,[1,0,2]) # 变成n_step在前面，后面要按n_step分割
    x = tf.reshape(x,[-1,n_input])  #变成二维的，n_step*batch_size
    x = tf.split(x,n_steps)  #分割成n_step个

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)

    output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm_fw_cell,
        lstm_bw_cell,
        x,
        dtype=tf.float32)
    return tf.matmul(output[-1],weights) + biases

#生成一个BiRNN
pred = BiRNN(x,weights,biases)
#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
#优化器
optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))  #返回最大值的index 然后做对比
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size,n_steps,n_input)) #用的python的reshape
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % display_step ==0:
            acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            #acc,loss = sess.run([accuracy,cost], feed_dict={x: batch_x, y: batch_y})
            print("Iter "+str(step*batch_size) + ", Minibatach loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimazation Finished")


    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label = mnist.test.labels[:test_len]
    print("testing Accuracy:" ,
      sess.run(accuracy,feed_dict={x:batch_x,y:batch_y}))
