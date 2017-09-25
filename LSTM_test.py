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


trainingLabel = np.zeros([9,20])
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

use projection layer get array [9 * 6 * 200]
projection matrix [5000, 200]

for each word in a sentence (which includes 6 words) calculate the lstm

use the previous output as the current status

after finishing the calculation add the results which leads to [9 * 200]

calculate the matrix multiplication [1200 * 200]
hidden matrix [1200 * 200]
hidden bias [200 * 1]

calculate cross entropy
'''
weights = {}
bias = {}

weights['projection'] = tf.Variable(tf.random_normal([5000,200]))
weights['hidden'] = tf.Variable(tf.random_normal([200, 20]))
bias['hidden'] = tf.Variable(tf.random_normal([20]))


def multilayerLSTMNet( sequenceBatch, batch_size ):
    
    outputForward = []
    outputBackward = []
    concatOutput = []
    lstm = tf.contrib.rnn.BasicLSTMCell(200)
    #initial state
    stateBackward = stateForward = tf.zeros([batch_size, lstm.state_size])

    concatVector = tf.nn.embedding_lookup(weights['projection'], sequenceBatch)

    for i in range(6):
        outputSlice, stateForward = lstm(concatVector[:,i,:], stateForward)
        outputForward.append(outputSlice)

    for i in range(6):
        outputSlice, stateBackward = lstm(concatVector[:,5 - i,:], stateBackward)
        outputBackward.append(outputSlice)

    for i in range(6):
        concatOutput.append(tf.add(outputForward[i] + outputBackward[5 - i]))

    readyToProcess = concatOutput[0]
    out = tf.add(tf.matmul(readyToProcess, weights['projection']),bias['hidden'])
    return out



sess = tf.Session()
sequence = tf.placeholder(tf.int32, [None, 6])
probability = tf.placeholder("float", [None, 20])
pred = multilayerLSTMNet(sequence, 9)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=probability,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.02).minimize(cost)
def trainingBatch(self, sequenceBatch, batch_probabilityClass):
        _, c = self.sess.run([optimizer, cost], feed_dict={sequence: sequenceBatch,
                                probability: batch_probabilityClass})
        return c


init = tf.global_variables_initializer();



sess.run(init)

for i in range(20):
    cost = trainingBatch(trainingSentencesBatch, trainingLabel)
    print(cost)