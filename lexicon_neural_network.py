import tensorflow as tf
import para
import math as mt
import numpy as np
import os.path

class TraditionalLexiconNet:
    def __init__(self, continue_pre = 0):
        #parameter
        self.weights = {}
        self.biases = {}
        self.netPara = para.Para.LexiconNeuralNetwork()


        #network
        parameter = para.Para()
        self.networkPathPrefix = parameter.GetNetworkStoragePath()

        # test for file exists
        testPath =  self.networkPathPrefix + 'lexicon_weight_projection.npy'
        if os.path.exists(testPath) == False:
            print('previous does not exist, start with random')
            continue_pre = 0


        if (continue_pre == 0):
            self.weights['projection'] = tf.Variable(tf.random_normal(self.netPara.GetProjectionLayer()))
            self.weights['hidden1'] = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer1st()))
            self.weights['hidden2'] = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer2nd()))
            self.weights['out'] = tf.Variable(tf.random_normal(self.netPara.GetOutputLayer()))
            self.biases['bHidden1'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer1st()[1]]))
            self.biases['bHidden2'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer2nd()[1]]))
            self.biases['out'] = tf.Variable(tf.random_normal([self.netPara.GetOutputLayer()[1]]))
        else:
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_weight_projection.npy')
            self.weights['projection'] = tf.Variable(savedMatrix)           
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_weight_hidden1.npy')
            self.weights['hidden1'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_weight_hidden2.npy')
            self.weights['hidden2'] = tf.Variable(savedMatrix)
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_weight_out.npy')
            self.weights['out'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_bias_bHidden1.npy')
            self.biases['bHidden1'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_bias_bHidden2.npy')
            self.biases['bHidden2'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_bias_out.npy')
            self.biases['out'] = tf.Variable(savedMatrix)

        
        # placeholder
        # for common words
        self.sess = tf.Session()
        self.sequence = tf.placeholder(tf.int32, [None, self.netPara.GetInputWordNum()])
        self.probabilityClass = tf.placeholder("float", [None, self.netPara.GetLabelSize()])
        self.pred, self.middle = self.multilayer_perceptron(self.sequence, self.netPara.GetInputWordNum())
        self.calculatedProb = tf.nn.softmax(self.pred)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probabilityClass,logits=self.pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.netPara.GetLearningRate()).minimize(self.cost)
        self.init = tf.global_variables_initializer();

        
        #initialize
        self.sess.run(self.init)

    def saveMatrixToFile(self):
        saveMatrix = self.sess.run(self.weights['projection'])
        np.save(self.networkPathPrefix + 'lexicon_weight_projection', saveMatrix)
        saveMatrix = self.sess.run(self.weights['hidden1'])
        np.save(self.networkPathPrefix + 'lexicon_weight_hidden1', saveMatrix)
        saveMatrix = self.sess.run(self.weights['hidden2'])
        np.save(self.networkPathPrefix + 'lexicon_weight_hidden2', saveMatrix)
        saveMatrix = self.sess.run(self.weights['out'])
        np.save(self.networkPathPrefix + 'lexicon_weight_out', saveMatrix)
        saveMatrix = self.sess.run(self.biases['bHidden1'])
        np.save(self.networkPathPrefix + 'lexicon_bias_bHidden1', saveMatrix)
        saveMatrix = self.sess.run(self.biases['bHidden2'])
        np.save(self.networkPathPrefix + 'lexicon_bias_bHidden2', saveMatrix)
        saveMatrix = self.sess.run(self.biases['out'])
        np.save(self.networkPathPrefix + 'lexicon_bias_out', saveMatrix)



    def multilayer_perceptron(self, sourceTarget, num):

        generateVector = tf.gather(self.weights['projection'], sourceTarget)
        unstackVector = tf.unstack(generateVector, None, 1)
        concatVector = tf.concat(unstackVector, 1)
        #concatVector = tf.nn.embedding_lookup(self.weights['projection'], sourceTarget)
        #concatVector = tf.reshape(concatVector ,[-1,1600])
        hiddenLayer1 = tf.add(tf.matmul(concatVector, self.weights['hidden1']), self.biases['bHidden1'])
        hiddenLayer1 = tf.nn.tanh(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, self.weights['hidden2']), self.biases['bHidden2'])
        hiddenLayer2 = tf.nn.tanh(hiddenLayer2)

        out = tf.add(tf.matmul(hiddenLayer2, self.weights['out']),self.biases['out'])
        return out, hiddenLayer2

    def networkPrognose(self, sourceTarget, lexiconLabel):
        self.output, middleOutput = self.sess.run([self.calculatedProb, self.middle],feed_dict={self.sequence : sourceTarget})
        outProbability = []

        for i in range(len(lexiconLabel)):
            outProbability.append(self.output[i][lexiconLabel[i]])
        return outProbability

    def trainingBatch(self, batch_sequence, batch_probabilityClass):
        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.sequence: batch_sequence,
                                self.probabilityClass: batch_probabilityClass})
        return c


class LSTMLexiconNet:
    def __init__(self, continue_pre = 0):
        # parameter
        self.weights = {}
        self.biases = {}
        self.netPara = para.Para.LexiconNeuralNetwork('lstm')


        # network
        parameter = para.Para()
        self.networkPathPrefix = parameter.GetNetworkStoragePath()

        # test for file exists
        testPath =  self.networkPathPrefix + 'lexicon_weight_projection.npy'
        if os.path.exists(testPath) == False:
            print('previous does not exist, start with random')
            continue_pre = 0
        # basic parameters
        self.sourceNum = 0
        self.targetNum = 0


        if (continue_pre == 0):
            self.weights['projection'] = tf.Variable(tf.random_normal(self.netPara.GetProjectionLayer()))
            self.weights['hidden1'] = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer1st()))
            self.weights['hidden2'] = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer2nd()))
            self.weights['out'] = tf.Variable(tf.random_normal(self.netPara.GetOutputLayer()))
            self.biases['bHidden1'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer1st()[1]]))
            self.biases['bHidden2'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer2nd()[1]]))
            self.biases['out'] = tf.Variable(tf.random_normal([self.netPara.GetOutputLayer()[1]]))
        else:
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_weight_projection.npy')
            self.weights['projection'] = tf.Variable(savedMatrix)           
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_weight_hidden1.npy')
            self.weights['hidden1'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_weight_hidden2.npy')
            self.weights['hidden2'] = tf.Variable(savedMatrix)
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_weight_out.npy')
            self.weights['out'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_bias_bHidden1.npy')
            self.biases['bHidden1'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_bias_bHidden2.npy')
            self.biases['bHidden2'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'lexicon_bias_out.npy')
            self.biases['out'] = tf.Variable(savedMatrix)

        
        # placeholder
        # for common words
        self.sess = tf.Session()
        self.sentence = tf.placeholder(tf.int32, [None])
        self.sourceNumPlace = tf.placeholder(tf.int32)
        self.targetNumPlace = tf.placeholder(tf.int32)
        self.probability = tf.placeholder("float", [None, self.netPara.GetLabelSize()])
        self.pred = self.multilayerLSTMNetForOneSentencePlaceholder(self.sentence, self.sourceNumPlace, self.targetNumPlace)
        self.calculatedProb = tf.nn.softmax(self.pred)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probability,logits=self.pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.netPara.GetLearningRate()).minimize(self.cost)
        self.init = tf.global_variables_initializer();

        
        #initialize
        self.sess.run(self.init)


    def saveMatrixToFile(self):
        saveMatrix = self.sess.run(self.weights['projection'])
        np.save(self.networkPathPrefix + 'lexicon_weight_projection', saveMatrix)
        saveMatrix = self.sess.run(self.weights['hidden1'])
        np.save(self.networkPathPrefix + 'lexicon_weight_hidden1', saveMatrix)
        saveMatrix = self.sess.run(self.weights['hidden2'])
        np.save(self.networkPathPrefix + 'lexicon_weight_hidden2', saveMatrix)
        saveMatrix = self.sess.run(self.weights['out'])
        np.save(self.networkPathPrefix + 'lexicon_weight_out', saveMatrix)
        saveMatrix = self.sess.run(self.biases['bHidden1'])
        np.save(self.networkPathPrefix + 'lexicon_bias_bHidden1', saveMatrix)
        saveMatrix = self.sess.run(self.biases['bHidden2'])
        np.save(self.networkPathPrefix + 'lexicon_bias_bHidden2', saveMatrix)
        saveMatrix = self.sess.run(self.biases['out'])
        np.save(self.networkPathPrefix + 'lexicon_bias_out', saveMatrix)


    def multilayerLSTMNetForOneSentencePlaceholder(self, sequence, _sourceNum, _targetNum):

        _concatOutput = tf.zeros([0,400])

        cell = tf.contrib.rnn.BasicLSTMCell(200, forget_bias=0.0, state_is_tuple=True, reuse=None)
        #initial state
        zeroState  = cell.zero_state(1, tf.float32)
        concatVector = tf.nn.embedding_lookup(self.weights['projection'], [sequence])
        _stateC = zeroState[0]
        _stateH = zeroState[1]
        i0 = tf.constant(0)
        j0 = tf.constant(0)
        _output = tf.zeros([0,200])
        # source forward part
        def sourceForwardBody(i, sourceNum, stateC, stateH, output):

            state = tf.contrib.rnn.LSTMStateTuple(stateC, stateH)
            outputSlice, state = cell(concatVector[:,i,:], state)
            output = tf.concat([output, outputSlice],0)
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
            output = tf.concat([outputSlice, output],0)
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
            output = tf.concat([output, outputSlice],0)
            stateC = state[0]
            stateH = state[1]
            i = tf.add(i,1)
            return i, targetNum, stateC, stateH, output
        def targetForwardCond(i, targetNum, stateC, stateH, output):
            return tf.less(i, targetNum)



        with tf.variable_scope("RNNLexicon"):

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

        # sample generation part, which means add the source forward and back 
        # back forward, then concat it to target forward
        def genSourceBody(i, j, sourceNum, output):
            item = tf.concat([tf.add(outputSourceForward[j, :], outputSourceBackward[j, :]), outputTargetForward[i,:]], 0)
            item = tf.reshape(item, [-1]);
            output = tf.concat([output, [item]], 0)
            j = tf.add(j, 1)
            return i, j, sourceNum, output
        def genSourceCond(i, j, sourceNum, output):
            return tf.less(j, sourceNum)

        def genTargetBody(i, targetNum, sourceNum, output):
            sourceLoop = tf.while_loop(
                genSourceCond,
                genSourceBody,
                loop_vars = [i, j0, sourceNum, output],
                shape_invariants = [
                i.get_shape(),
                j0.get_shape(),
                sourceNum.get_shape(),
                tf.TensorShape([None, 400])])
            output = sourceLoop[3]
            i = tf.add(i,1)
            return i, targetNum, sourceNum, output

        def genTargetCond(i, targetNum, sourceNum, output):
            return tf.less(i,targetNum)
        targetLoop = tf.while_loop(
            genTargetCond,
            genTargetBody,
            loop_vars = [i0, _targetNum, _sourceNum, _concatOutput],
            shape_invariants = [
            i0.get_shape(),
            _targetNum.get_shape(),
            _sourceNum.get_shape(),
            tf.TensorShape([None, 400])])

        readyToProcess = targetLoop[3]

        hiddenLayer1 = tf.add(tf.matmul(readyToProcess, self.weights['hidden1']), self.biases['bHidden1'])
        hiddenLayer1 = tf.nn.tanh(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, self.weights['hidden2']), self.biases['bHidden2'])
        hiddenLayer2 = tf.nn.tanh(hiddenLayer2)

        out = tf.add(tf.matmul(hiddenLayer2, self.weights['out']),self.biases['out'])

        return out
    def networkPrognose(self, sentence, lexiconLabel, _sourceNum, _targetNum):
        self.output = self.sess.run([self.calculatedProb],feed_dict={self.sentence : sentence,
            self.sourceNumPlace : _sourceNum,
            self.targetNumPlace : _targetNum})
        outProbability = []

        for i in range(len(lexiconLabel)):
            outProbability.append(self.output[i][lexiconLabel[i]])
        return outProbability

    def trainingBatch(self, sentence, batch_probability, _sourceNum, _targetNum):
        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.sentence: sentence,
                                self.probability: batch_probability,
                                self.sourceNumPlace : _sourceNum,
                                self.targetNumPlace : _targetNum})
        return c
