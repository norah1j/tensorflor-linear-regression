# Linear Regression with TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_FILE = 'gdp2life.txt'

def read_data(filename):
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    gdp = [float(line[1]) for line in data]    
    lifes = [float(line[2]) for line in data]
    data = list(zip(gdp, lifes))
    n_samples = len(data)
    # surpress scientific notation
    np.set_printoptions(suppress=True)
    data = np.asarray(data, dtype=np.float32)
    print(data)
    print(n_samples)
    return data, n_samples
                                    

data, n_samples = read_data(DATA_FILE)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))
Y_predicted = w * X + b
loss = tf.square(Y - Y_predicted, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0003).minimize(loss)
writer = tf.summary.FileWriter('./graphs/linear_reg', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    # train model
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, mloss =  sess.run([optimizer, loss], feed_dict={X: x, Y:y})
            total_loss += mloss

        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
    writer.close()
    w_out, b_out = sess.run([w,b])

# make plot
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='predicted data')
plt.legend()
plt.show()


