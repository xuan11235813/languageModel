import tensorflow as tf
import para
import math as mt
import numpy as np
import os.path


class TraditionalAlignmentNet:
    def __init__(self, continue_pre = 0):
        #parameter
        self.weights = {}
        self.biases = {}
        self.netPara = para.Para.AlignmentNeuralNetwork()

        parameter = para.Para()
        self.networkPathPrefix = parameter.GetNetworkStoragePath()

        # test for file exists
        testPath =  self.networkPathPrefix + 'alignment_weight_projection.npy'
        if os.path.exists(testPath) == False:
            print('previous does not exist, start with random')
            continue_pre = 0

        if (continue_pre == 0):
            self.weights['projection'] = tf.Variable(tf.random_normal(self.netPara.GetProjectionLayer()))
            self.weights['hidden1'] = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer1st()))
            self.weights['hidden2'] = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer2nd()))
            self.weights['out'] = tf.Variable(tf.random_normal(self.netPara.GetJumpLayer()))
            self.biases['bHidden1'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer1st()[1]]))
            self.biases['bHidden2'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer2nd()[1]]))
            self.biases['out'] = tf.Variable(tf.random_normal([self.netPara.GetJumpLayer()[1]]))

        else:
            savedMatrix = np.load(self.networkPathPrefix + 'alignment_weight_projection.npy')
            self.weights['projection'] = tf.Variable(savedMatrix)           
            savedMatrix = np.load(self.networkPathPrefix + 'alignment_weight_hidden1.npy')
            self.weights['hidden1'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'alignment_weight_hidden2.npy')
            self.weights['hidden2'] = tf.Variable(savedMatrix)
            savedMatrix = np.load(self.networkPathPrefix + 'alignment_weight_out.npy')
            self.weights['out'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'alignment_bias_bHidden1.npy')
            self.biases['bHidden1'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'alignment_bias_bHidden2.npy')
            self.biases['bHidden2'] = tf.Variable(savedMatrix)            
            savedMatrix = np.load(self.networkPathPrefix + 'alignment_bias_out.npy')
            self.biases['out'] = tf.Variable(savedMatrix)
        #network
        
        #placeholder
        self.sess = tf.Session()
        self.sequence = tf.placeholder(tf.int32, [None, self.netPara.GetInputWordNum()])
        self.probability = tf.placeholder("float", [None, self.netPara.GetJumpLabelSize()])
        self.pred = self.multilayer_perceptron(self.sequence, self.netPara.GetInputWordNum())
        self.calculatedProb = tf.nn.softmax(self.pred)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probability,logits=self.pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.netPara.GetLearningRate()).minimize(self.cost)
        self.init = tf.global_variables_initializer();
        
        #initialize
        self.sess.run(self.init)

    def saveMatrixToFile(self):
        saveMatrix = self.sess.run(self.weights['projection'])
        np.save(self.networkPathPrefix + 'alignment_weight_projection', saveMatrix)
        saveMatrix = self.sess.run(self.weights['hidden1'])
        np.save(self.networkPathPrefix + 'alignment_weight_hidden1', saveMatrix)
        saveMatrix = self.sess.run(self.weights['hidden2'])
        np.save(self.networkPathPrefix + 'alignment_weight_hidden2', saveMatrix)
        saveMatrix = self.sess.run(self.weights['out'])
        np.save(self.networkPathPrefix + 'alignment_weight_out', saveMatrix)
        saveMatrix = self.sess.run(self.biases['bHidden1'])
        np.save(self.networkPathPrefix + 'alignment_bias_bHidden1', saveMatrix)
        saveMatrix = self.sess.run(self.biases['bHidden2'])
        np.save(self.networkPathPrefix + 'alignment_bias_bHidden2', saveMatrix)
        saveMatrix = self.sess.run(self.biases['out'])
        np.save(self.networkPathPrefix + 'alignment_bias_out', saveMatrix)

    def multilayer_perceptron(self, sourceTarget, num):

        generateVector = tf.gather(self.weights['projection'], sourceTarget)
        unstackVector = tf.unstack(generateVector, None, 1)
        concatVector = tf.concat(unstackVector, 1)
        hiddenLayer1 = tf.add(tf.matmul(concatVector, self.weights['hidden1']), self.biases['bHidden1'])
        hiddenLayer1 = tf.nn.sigmoid(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, self.weights['hidden2']), self.biases['bHidden2'])
        hiddenLayer2 = tf.nn.sigmoid(hiddenLayer2)

        outLayer = tf.add(tf.matmul(hiddenLayer2, self.weights['out']),self.biases['out'])
        return outLayer

    def networkPrognose(self, sourceTarget, sourceTargetInitial):
        outInitial = self.sess.run(self.calculatedProb, feed_dict = {self.sequence : [sourceTargetInitial]})
        out = self.sess.run(self.calculatedProb,feed_dict={self.sequence : sourceTarget})
        return out, outInitial

    def trainingBatch(self, batch_sequence, batch_probability):
        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.sequence: batch_sequence,
                                self.probability: batch_probability})
        return c

    def trainingInitialState(self, sequence_initial, probability_initial):

        batch_sequence = []
        batch_probability = []
        for i in range(5):
            batch_sequence.append(sequence_initial)
            batch_probability.append(probability_initial)

        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.sequence: batch_sequence,
                                self.probability: batch_probability})

        return c

    def trainingBatchWithInitial( self, batch_sequence, batch_probability, sequence_initial, probability_initial):
        batch_probability = np.array(batch_probability)
        for i in range(5):
            batch_sequence.append(sequence_initial)
            batch_probability = np.append(batch_probability, np.array([probability_initial]), axis = 0)
            #batch_probability.append(probability_initial)

        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.sequence: batch_sequence,
                                self.probability: batch_probability})
        return c


class LSTMAlignmentNet:
    def __init__(self, continue_pre = 0):
        # parameter
        self.weights = {}
        self.biases = {}
        self.parameter = para.Para()
        self.netPara = para.Para.AlignmentNeuralNetwork('lstm')
        self.sourceTargetBias = self.parameter.GetTargetSourceBias()
        self.projOutDim = self.netPara.GetProjectionLayer()[1]
        self.readyToProcessDim = 2 * self.projOutDim

        parameter = para.Para()
        self.networkPathPrefix = parameter.GetNetworkStoragePath()
        self.batchSize = parameter.GetLSTMBatchSize()
         # test for file exists
        testPath =  self.networkPathPrefix + 'alignmentModel.meta'
        if os.path.exists(testPath) == False:
            print('previous does not exist, start with random')
            continue_pre = 0
        self.weights_projection_source = tf.Variable(tf.random_normal(self.netPara.GetProjectionLayer()), name="alignment_weight_projection_source")
        self.weights_projection_target= tf.Variable(tf.random_normal(self.netPara.GetProjectionLayer()), name="alignment_weight_projection_target")
        self.weights_hidden1 = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer1st()), name="alignment_weight_hidden1")
        self.weights_hidden2 = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer2nd()), name="alignment_weight_hidden2")
        self.weights_out = tf.Variable(tf.random_normal(self.netPara.GetJumpLayer()), name="alignment_weight_out")
        self.biases_bHidden1 = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer1st()[1]]), name="alignment_bias_bHidden1")
        self.biases_bHidden2 = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer2nd()[1]]), name="alignment_bias_bHidden2")
        self.biases_out = tf.Variable(tf.random_normal([self.netPara.GetJumpLayer()[1]]), name="alignment_bias_out")
        
        with tf.variable_scope("RNNAlignment"):
            self.cell_fw = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
            self.cell_bw = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
            self.cell_target = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
            self.initial_state_fw  = self.cell_fw.zero_state(1, tf.float32)
            self.initial_state_bw  = self.cell_bw.zero_state(1, tf.float32)
            self.initial_state_target = self.cell_target.zero_state(1, tf.float32)
            self.cell = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
            self.zeroState  = self.cell.zero_state(2, tf.float32)
        #network
        
        # placeholder
        # for common words

        self.sess = tf.Session()
        
        self.sequenceBatch = tf.placeholder(tf.int32, [None, None])
        self.sourceNumPlace = tf.placeholder(tf.int32)
        self.targetNumPlace = tf.placeholder(tf.int32)
        self.sourceTargetPlace = tf.placeholder(tf.int32, [2])
        self.probability = tf.placeholder("float", [None, self.netPara.GetJumpLabelSize()])
        self.learningRate = tf.placeholder("float")
        #self.pred = self.multilayerLSTMNetForOneSentencePlaceholder(self.sequenceBatch, self.sourceTargetPlace)
        self.pred = self.multilayerLSTMNetModern(self.sequenceBatch, self.sourceTargetPlace)
        self.calculatedProb = tf.nn.softmax(self.pred)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probability,logits=self.pred))
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate= self.learningRate).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learningRate).minimize(self.cost)
        # only for zero input
        self.predInit = self.getLSTMInitial()
        self.calculatedProbInit = tf.nn.softmax(self.predInit)
        self.costInit = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probability,logits=self.predInit))
        #self.optimizerInit = tf.train.GradientDescentOptimizer(learning_rate= self.learningRate).minimize(self.costInit)
        self.optimizerInit = tf.train.AdamOptimizer(learning_rate= self.learningRate).minimize(self.costInit)
        self.init = tf.global_variables_initializer();
        
        # for translation
        self.translationPred = self.multilayerLSTMNetTranslationPredict(self.sequenceBatch, self.sourceTargetPlace)
        self.translationProb = tf.nn.softmax(self.translationPred)
        #initialize
        self.saver = tf.train.Saver()
        if (continue_pre == 1):
            print('read from trained file')
            self.saver.restore(self.sess, self.networkPathPrefix + 'alignmentModel')
        else:
            self.sess.run(self.init)
 


    def saveMatrixToFile(self):
        self.saver.save(self.sess, self.networkPathPrefix + 'alignmentModel')

    def multilayerLSTMNetModern(self, sequence_batch, _sourcetargetNum):


        _concatOutput = tf.zeros([0,self.readyToProcessDim])
        i0 = tf.constant(0)
        j0 = tf.constant(0)
        _output = tf.zeros([0,self.projOutDim])
        seqIndexSource, seqIndexTarget = tf.split(sequence_batch, _sourcetargetNum, 1)
        seqSource = tf.nn.embedding_lookup(self.weights_projection_source, seqIndexSource)
        seqTarget = tf.nn.embedding_lookup(self.weights_projection_target, seqIndexTarget)
        
        with tf.variable_scope("RNNAlignment"):

            #seqSource, seqTarget = tf.split(concatVector, _sourcetargetNum, 1)


            (outputSource, _) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, seqSource,
                        tf.stack([_sourcetargetNum[0]]), self.initial_state_fw, self.initial_state_bw)
            (outputTargetForward, _) = tf.nn.dynamic_rnn( self.cell_target, seqTarget, 
                        tf.stack([_sourcetargetNum[1]]), self.initial_state_target )

        outputSourceForward = outputSource[0]
        outputSourceBackward = outputSource[1]


        def genSourceBody(i, j, sourceNum, output):
            
            item = tf.concat([tf.add(outputSourceForward[:,j, :], outputSourceBackward[:,j, :]), outputTargetForward[:,i,:]], 1)
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
                tf.TensorShape([None, self.readyToProcessDim])])
            output = sourceLoop[3]
            i = tf.add(i,1)
            return i, targetNum, sourceNum, output

        def genTargetCond(i, targetNum, sourceNum, output):
            return tf.less(i,targetNum)

        with tf.variable_scope("RNNAlignment"):
            targetLoop = tf.while_loop(
                genTargetCond,
                genTargetBody,
                loop_vars = [i0, _sourcetargetNum[1], _sourcetargetNum[0], _concatOutput],
                shape_invariants = [
                i0.get_shape(),
                _sourcetargetNum[0].get_shape(),
                _sourcetargetNum[0].get_shape(),
                tf.TensorShape([None, self.readyToProcessDim])])

        readyToProcess = targetLoop[3]

        hiddenLayer1 = tf.add(tf.matmul(readyToProcess, self.weights_hidden1), self.biases_bHidden1)
        hiddenLayer1 = tf.nn.tanh(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, self.weights_hidden2), self.biases_bHidden2)        
        hiddenLayer2 = tf.nn.tanh(hiddenLayer2)

        out = tf.add(tf.matmul(hiddenLayer2, self.weights_out),self.biases_out)

        return out

        '''
    def multilayerLSTMNetForOneSentencePlaceholder(self, sequence_batch, _sourceTargetNum):

        _sourceNum = _sourceTargetNum[0]
        _targetNum = _sourceTargetNum[1]

        # loops only equals targetNum - 1
        #_targetNum = tf.subtract(_targetNum, 1)
        _concatOutput = tf.zeros([0,self.readyToProcessDim])

        cell = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
        #initial state
        zeroState  = cell.zero_state(self.batchSize, tf.float32)
        concatVector = tf.nn.embedding_lookup(self.weights_projection, sequence_batch)

        # here if you change batchSize from 1 to other value here maybe something wrong.
        _stateC = zeroState[0]
        _stateH = zeroState[1]
        i0 = tf.constant(0)
        j0 = tf.constant(0)
        _output = tf.zeros([0,self.projOutDim])

        with tf.variable_scope("RNNAlignment"):
            _, state = cell(concatVector[:,0,:], zeroState)
        # source forward part
        def sourceForwardBody(i, sourceNum, stateC, stateH, output):
            tf.get_variable_scope().reuse_variables()
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
            tf.get_variable_scope().reuse_variables()
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
            tf.get_variable_scope().reuse_variables()
            state = tf.contrib.rnn.LSTMStateTuple(stateC, stateH)
            outputSlice, state = cell(concatVector[:,tf.add(_sourceNum, i),:], state)
            output = tf.concat([output, outputSlice],0)
            stateC = state[0]
            stateH = state[1]
            i = tf.add(i,1)
            return i, targetNum, stateC, stateH, output
        def targetForwardCond(i, targetNum, stateC, stateH, output):
            return tf.less(i, targetNum)



        with tf.variable_scope("RNNAlignment"):

            sourceForwardLoop = tf.while_loop(
                sourceForwardCond,
                sourceForwardBody,
                loop_vars = [i0, _sourceNum, _stateC, _stateH, _output],
                shape_invariants = [
                i0.get_shape(),
                _sourceNum.get_shape(),
                _stateC.get_shape(),
                _stateH.get_shape(),
                tf.TensorShape([None, self.projOutDim])])
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
                tf.TensorShape([None, self.projOutDim])])
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
                tf.TensorShape([None, self.projOutDim])])
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
                tf.TensorShape([None, self.readyToProcessDim])])
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
            tf.TensorShape([None, self.readyToProcessDim])])

        readyToProcess = targetLoop[3]

        hiddenLayer1 = tf.add(tf.matmul(readyToProcess, self.weights_hidden1), self.biases_bHidden1)
        hiddenLayer1 = tf.nn.tanh(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, self.weights_hidden2), self.biases_bHidden2)
        hiddenLayer2 = tf.nn.tanh(hiddenLayer2)

        out = tf.add(tf.matmul(hiddenLayer2, self.weights_out),self.biases_out)

        return out

        '''
    def multilayerLSTMNetTranslationPredict(self, sequence_batch, _sourcetargetNum):

        # here should be noticed that the target number should not be minused 1
        # we are going to predict


        _concatOutput = tf.zeros([0,self.readyToProcessDim])
        i0 = tf.constant(0)
        j0 = tf.constant(0)
        _output = tf.zeros([0,self.projOutDim])
        seqIndexSource, seqIndexTarget = tf.split(sequence_batch, _sourcetargetNum, 1)
        seqSource = tf.nn.embedding_lookup(self.weights_projection_source, seqIndexSource)
        seqTarget = tf.nn.embedding_lookup(self.weights_projection_target, seqIndexTarget)
        
        with tf.variable_scope("RNNAlignment"):

            #seqSource, seqTarget = tf.split(concatVector, _sourcetargetNum, 1)


            (outputSource, _) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, seqSource,
                        tf.stack([_sourcetargetNum[0]]), self.initial_state_fw, self.initial_state_bw)
            (outputTargetForward, _) = tf.nn.dynamic_rnn( self.cell_target, seqTarget, 
                        tf.stack([_sourcetargetNum[1]]), self.initial_state_target )
        '''
        _concatOutput = tf.zeros([0,self.readyToProcessDim])
        i0 = tf.constant(0)
        j0 = tf.constant(0)
        _output = tf.zeros([0,self.projOutDim])
        concatVector = tf.nn.embedding_lookup(self.weights_projection, sequence_batch)
        with tf.variable_scope("RNNAlignment"):
            seqSource, seqTarget = tf.split(concatVector, _sourcetargetNum, 1)


            (outputSource, _) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, seqSource,
                        tf.stack([_sourcetargetNum[0]]), self.initial_state_fw, self.initial_state_bw)
            (outputTargetForward, _) = tf.nn.dynamic_rnn( self.cell_target, seqTarget, 
                        tf.stack([_sourcetargetNum[1]]), self.initial_state_target )
    '''

        outputSourceForward = outputSource[0]
        outputSourceBackward = outputSource[1]
        
        combinedArray = tf.concat([tf.add(outputSourceForward[:,_sourcetargetNum[0], :], 
            outputSourceBackward[:,_sourcetargetNum[0], :]), 
            outputTargetForward[:,_sourcetargetNum[1],:]], 1)
        # item = tf.reshape(item, [-1]);

        readyToProcess = combinedArray

        hiddenLayer1 = tf.add(tf.matmul(readyToProcess, self.weights_hidden1), self.biases_bHidden1)
        hiddenLayer1 = tf.nn.tanh(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, self.weights_hidden2), self.biases_bHidden2)        
        hiddenLayer2 = tf.nn.tanh(hiddenLayer2)

        out = tf.add(tf.matmul(hiddenLayer2, self.weights_out),self.biases_out)

        return out
        
    def getLSTMInitial(self):
        with tf.variable_scope("RNNAlignment"):
        #cell = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
        #zeroState  = cell.zero_state(2, tf.float32)
            sourceZero = tf.nn.embedding_lookup(self.weights_projection_source,[[0]])
            targetZero = tf.nn.embedding_lookup(self.weights_projection_target,[[self.sourceTargetBias]])
            concatVector = tf.concat([sourceZero, targetZero],0)       
            outputSlice, _ = self.cell(concatVector[:,0,:], self.zeroState)
        readyToProcess = [tf.reshape(outputSlice, [-1])]

        hiddenLayer1 = tf.add(tf.matmul(readyToProcess, self.weights_hidden1), self.biases_bHidden1)
        hiddenLayer1 = tf.nn.tanh(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, self.weights_hidden2), self.biases_bHidden2)
        hiddenLayer2 = tf.nn.tanh(hiddenLayer2)

        out = tf.add(tf.matmul(hiddenLayer2, self.weights_out),self.biases_out)

        return out

    def networkPrognose(self, _sequenceBatch,  _sourceNum, _targetNum):
        outInitial = self.sess.run(self.calculatedProbInit, feed_dict = {})

        out = self.sess.run([self.calculatedProb],feed_dict={self.sequenceBatch : _sequenceBatch,
            self.sourceTargetPlace : [_sourceNum, _targetNum - 1]})


        return out[0], outInitial

    def trainingBatchWithInitial( self, _sequenceBatch, probability, probability_initial, _sourceNum, _targetNum, learningRate):
        probability = np.array(probability)

        _, c = self.sess.run([self.optimizerInit, self.costInit], feed_dict = {self.probability :[probability_initial],
                                                                            self.learningRate : learningRate})
        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.sequenceBatch: _sequenceBatch,
                                self.probability: probability,
                                self.sourceTargetPlace : [_sourceNum, _targetNum -1],
                                self.learningRate : learningRate})
        return c

    def networkTranslationPrognose(self, _sequenceBatch, _sourceNum, _targetNum):

        # here the parameters of sourceTargetPlace shall be considered again
        out = self.sess.run([self.translationProb],feed_dict={self.sequenceBatch : _sequenceBatch,
            self.sourceTargetPlace : [_sourceNum, _targetNum]})

        return out[0]
    def networkTranslationInitial(self):
        outInitial = self.sess.run(self.calculatedProbInit, feed_dict = {})
        return outInitial

