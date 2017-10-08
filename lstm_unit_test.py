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

_trainingSentence = [32,55,72,91,23,12]
_trainingSource = [32,55,72]
_trainingTarget = [91,23,12]

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
weights['hiddenSequence'] = tf.Variable(tf.random_normal([400, 20]))
bias['hidden'] = tf.Variable(tf.random_normal([20]))


#sentences batch with same length.
def multilayerLSTMNet( sequenceBatch, batch_size ):
    
    outputForward = []
    outputBackward = []
    concatOutput = []
    cell = tf.contrib.rnn.BasicLSTMCell(200, forget_bias=0.0, state_is_tuple=True, reuse=None)
    # for multiple lstm
    # cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range( layers )], state_is_tuple = True)


    #initial state
    stateBackward = stateForward = cell.zero_state(batch_size, tf.float32)

    concatVector = tf.nn.embedding_lookup(weights['projection'], sequenceBatch)
    with tf.variable_scope("RNN"):
        for i in range(6):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            outputSlice, stateForward = cell(concatVector[:,i,:], stateForward)
            outputForward.append(outputSlice)

        for i in range(6):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            outputSlice, stateBackward = cell(concatVector[:,5 - i,:], stateBackward)
            outputBackward.append(outputSlice)

    for i in range(6):
        concatOutput.append(tf.add(outputForward[i] , outputBackward[5 - i]))

    readyToProcess = concatOutput[3]
    out = tf.add(tf.matmul(readyToProcess, weights['hidden']),bias['hidden'])
    return out




# input only one sentence with uncertain length
def multilayerLSTMNetForOneSentence(sequence, sourceNum, targetNum):

    outputSourceForward = []
    outputSourceBackward = []
    outputTargetForward = []
    concatOutput = []

    cell = tf.contrib.rnn.BasicLSTMCell(200, forget_bias=0.0, state_is_tuple=True, reuse=None)
    #initial state
    stateSourceBackward = stateSourceForward = stateTargetForward = cell.zero_state(1, tf.float32)
    print(stateSourceForward[0].get_shape())
    print(stateSourceForward[1].get_shape())
    concatVector = tf.nn.embedding_lookup(weights['projection'], [sequence])
    with tf.variable_scope("RNN"):
        for i in range(3):
            print(i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            outputSlice, stateSourceForward = cell(concatVector[:,i,:], stateSourceForward)
            outputSourceForward.append(outputSlice)

        for i in range(3):
            print(3 - i - 1)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            outputSlice, stateSourceBackward = cell(concatVector[:,3 - i - 1,:], stateSourceBackward)
            outputSourceBackward.insert(0, outputSlice)

        for i in range(3):
            print(3 + i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            outputSlice, stateTargetForward = cell(concatVector[:,3 + i,:], stateTargetForward)
            outputTargetForward.append(outputSlice)

    for i in range(3):
        for j in range(3):
            item = tf.concat( [tf.add(outputSourceForward[j] , outputSourceForward[j]), outputTargetForward[i] ] , 1)
            item = tf.reshape(item, [-1]);
            concatOutput.append(item)
    readyToProcess = tf.stack(concatOutput)
    out = tf.add(tf.matmul(readyToProcess, weights['hiddenSequence']),bias['hidden'])
    print('------------------end process--------------------')
    return out




# input only one sentence with uncertain length( placeholder )
def multilayerLSTMNetForOneSentencePlaceholder(sequence, _sourceNum, _targetNum):
    
    _outputSourceForward = tf.zeros([0,200])
    _outputSourceBackward = tf.zeros([0,200])
    _outputTargetForward = tf.zeros([0,200])
    _concatOutput = tf.zeros([0,400])

    cell = tf.contrib.rnn.BasicLSTMCell(200, forget_bias=0.0, state_is_tuple=True, reuse=None)
    #initial state
    zeroState  = cell.zero_state(1, tf.float32)
    concatVector = tf.nn.embedding_lookup(weights['projection'], [sequence])
    _stateC = zeroState[0]
    _stateH = zeroState[1]
    i0 = tf.constant(0)
    j0 = tf.constant(0)
    _output = tf.zeros([0,200])
    # source forward part
    def sourceForwardBody(i, sourceNum, stateC, stateH, output):

        state = tf.contrib.rnn.LSTMStateTuple(stateC, stateH)
        outputSlice, state = cell(concatVector[:,i,:], state)
        output = tf.concat([output, outputSlice])
        stateC = state[0]
        stateH = state[1]
        i = tf.add(i, 1)
        return i, sourceNum, stateC, stateH, output
    def sourceForwardCond(i, sourceNum, stateC, stateH, output):
        return  tf.less(i, sourceNum)

    # source backward part
    def sourceBackwardBody(i, sourceNum, stateC, stateH, output):
        state = tf.contrib.rnn.LSTMStateTuple(stateC, stateH)
        outputSlice, state = cell(concatVector[:,tf.subtract(sourceNum, tf.add(i,1)),:], state)
        output = tf.concat([outputSlice, output])
        stateC = state[0]
        stateH = state[1]
        i = tf.add(i,1)
        return i, sourceNum, stateC, stateH, output
    def sourceBackwardCond(i, sourceNum, stateC, stateH, output):
        return  tf.less(i, sourceNum)

    # target forward part
    def targetForwardBody(i, targetNum, stateC, stateH, output):
        state = tf.contrib.rnn.LSTMStateTuple(stateC, stateH)
        outputSlice, state = cell(concatVector[:,tf.add(_sourceNum, i),:], state)
        output = tf.concat([output, outputSlice])
        stateC = state[0]
        stateH = state[1]
        i = tf.add(i,1)
        return i, targetNum, stateC, stateH, output
    def targetForwardCond(i, targetNum, stateC, stateH, output):
        return tf.less(i, targetNum)
    with tf.variable_scope("RNN"):

        sourceForwardLoop = tf.while_loop(
            sourceForwardCond,
            sourceForwardBody,
            loop_vars = [i0, _sourceNum, _stateC, _stateH, _output],
            shape_invariants = [
            i0.get_shape(),
            _sourceNum.get_shape(),
            _stateC.get_shape(),
            _stateH.get_shape(),
            tf.TensorShape([None, 200])])
        outputSourceForward = sourceForwardLoop[4]

        sourceBackwardLoop = tf.while_loop(
            sourceForwardCond,
            sourceBackwardBody,
            loop_vars = [i0, _sourceNum, _stateC, _stateH, _output],
            shape_invariants = [
            i0.get_shape(),
            _sourceNum.get_shape(),
            _stateC.get_shape(),
            _stateH.get_shape(),
            tf.TensorShape([None, 200])])
        outputSourceBackward = sourceBackwardLoop[4]

        targetForwardLoop = tf.while_loop(
            targetForwardCond,
            targetForwardBody,
            loop_vars = [i0, _targetNum, _stateC, _stateH, _output],
            shape_invariants = [
            i0.get_shape(),
            _targetNum.get_shape(),
            _stateC.get_shape(),
            _stateH.get_shape(),
            tf.TensorShape([None, 200])])
        outputTargetForward = targetForwardLoop[4]
        
    for i in range(3):
        for j in range(3):
            item = tf.concat( [tf.add(outputSourceForward[j] , outputSourceForward[j]), outputTargetForward[i] ] , 1)
            item = tf.reshape(item, [-1]);
            concatOutput.append(item)
    readyToProcess = tf.stack(concatOutput)
    out = tf.add(tf.matmul(readyToProcess, weights['hiddenSequence']),bias['hidden'])
    print('------------------end process--------------------')
    return out


sess = tf.Session()
sequence = tf.placeholder(tf.int32, [None, None])
sentence = tf.placeholder(tf.int32, [None])
sourceNumPlace = tf.placeholder(tf.int32)
targetNumPlace = tf.placeholder(tf.int32)
_sourceNum = 3
_targetNum = 3
probability = tf.placeholder("float", [None, 20])
print(probability.get_shape())
#pred = multilayerLSTMNet(sequence, 9)
pred = multilayerLSTMNetForOneSentence( sentence, sourceNumPlace,sourceNumPlace )
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=probability,logits=pred))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=probability,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.02).minimize(cost)
def trainingBatch(sequenceBatch, batch_probabilityClass):
    _, c = sess.run([optimizer, cost], feed_dict={sequence: sequenceBatch, probability: batch_probabilityClass})
    return c
def trainingSentence(sequence, batch_probabilityClass):
    _, c = sess.run([optimizer, cost], feed_dict={sentence: sequence, probability: batch_probabilityClass,\
     sourceNumPlace: _sourceNum, targetNumPlace: _targetNum})
    return c

init = tf.global_variables_initializer();



sess.run(init)

for i in range(20): 
    costValue = trainingSentence(_trainingSentence, trainingLabel)
    print(costValue)

'''

for i in range(20): 
    costValue = trainingBatch(trainingSentencesBatch, trainingLabel)
    print(costValue)

'''