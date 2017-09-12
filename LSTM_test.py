import tensorflow as tf
import numpy as np


trainingSentencesBatch = [
                    [24,33,46,53,62,81],
                    [32,55,72,91,23,12],
                    [52,63,87,34,23,45],
                    [24,33,46,53,62,81],
                    [32,55,72,91,23,12],
                    [52,63,87,34,23,45],
                    [24,33,46,53,62,81],
                    [32,55,72,91,23,12],
                    [52,63,87,34,23,45]
                    ]


trainingLabel = np.zeros([9,200])
trainingLabel[0][3] = 1
trainingLabel[1][2] = 1
trainingLabel[2][1] = 1
trainingLabel[3][3] = 1
trainingLabel[4][2] = 1
trainingLabel[5][1] = 1
trainingLabel[6][3] = 1
trainingLabel[7][2] = 1
trainingLabel[8][1] = 1

'''
sentences: 6 words, batch size = 9 [9 * 6]

use projection layer and concatenate the obtained array [9 * 6 * 200] -> [9 * 1200]
projection matrix [5000, 200]

hidden network lstm, then use output to calculate the matrix multiplication [1200 * 200]
hidden matrix [1200 * 200]
hidden bias [200 * 1]

calculate cross entropy
'''


weights['projection'] = tf.Variable(tf.random_normal([5000,200]))
weights['hidden'] = tf.Variable(tf.random_normal([1200, 200]))

bias['hidden'] = tf.Variable(tf.random_normal(200))

lstm = tf.contrib.rnn.BasicLSTMCell(1200)
#initial state
state = tf.zeros([batch_size, lstm.state_size])


def multilayerLSTMNet( sequenceBatch, batch_size ):
    
    lstm = tf.contrib.rnn.BasicLSTMCell(1200)
    #initial state
    state = tf.zeros([batch_size, lstm.state_size])
    
    concatVector = tf.nn.egmbedding_lookup(weights['projection'], sequenceBatch)
    concatVector = tf.reshape(concatVector, [-1, 1200])
    output, state = lstm(concatVector, state)



sess = tf.Session()
sequence = tf.placeholder(tf.int32, [None, 6])
probability = tf.placeholder("float", [None, 200])
