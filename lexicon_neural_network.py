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

# should be noticed that it is really easy to change model to sentence batch
# take care of the state


class LSTMLexiconNet:
    def __init__(self, continue_pre = 0):
        # parameter
        self.weights = {}
        self.biases = {}
        self.netPara = para.Para.LexiconNeuralNetwork('lstm')
        self.projOutDim = self.netPara.GetProjectionLayer()[1]
        self.readyToProcessDim = 2 * self.projOutDim

        # network
        parameter = para.Para()
        self.networkPathPrefix = parameter.GetNetworkStoragePath()
        self.batchSize = parameter.GetLSTMBatchSize()
        # test for file exists
        testPath =  self.networkPathPrefix + 'lexiconModel.meta'
        if os.path.exists(testPath) == False:
            print('previous does not exist, start with random')
            continue_pre = 0
        # basic parameters
        self.sourceNum = 0
        self.targetNum = 0


        self.weights_projection_source = tf.Variable(tf.random_normal(self.netPara.GetProjectionLayer()), name="lexicon_weight_projection_source")
        self.weights_projection_target = tf.Variable(tf.random_normal(self.netPara.GetProjectionLayer()), name="lexicon_weight_projection_target")
        self.weights_hidden1 = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer1st()), name="lexicon_weight_hidden1")
        self.weights_hidden2 = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer2nd()), name="lexicon_weight_hidden2")
        self.weights_out = tf.Variable(tf.random_normal(self.netPara.GetOutputLayer()), name="lexicon_weight_out")
        self.biases_bHidden1 = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer1st()[1]]), name="lexicon_bias_bHidden1")
        self.biases_bHidden2 = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer2nd()[1]]), name="lexicon_bias_bHidden2")
        self.biases_out = tf.Variable(tf.random_normal([self.netPara.GetOutputLayer()[1]]), name="lexicon_bias_out")
 
        with tf.variable_scope("RNNLexicon"):
            self.cell_fw = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
            self.cell_bw = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
            self.cell_target = tf.contrib.rnn.BasicLSTMCell(self.projOutDim, forget_bias=0.0, state_is_tuple=True, reuse=None)
            self.initial_state_fw  = self.cell_fw.zero_state(1, tf.float32)
            self.initial_state_bw  = self.cell_bw.zero_state(1, tf.float32)
            self.initial_state_target = self.cell_target.zero_state(1, tf.float32)
        
        # placeholder
        # for common words
        #self.sess = sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()
        
        self.sequenceBatch = tf.placeholder(tf.int32, [None, None])
        self.sourceNumPlace = tf.placeholder(tf.int32)
        self.targetNumPlace = tf.placeholder(tf.int32)
        self.sourceTargetPlace = tf.placeholder(tf.int32, [2])
        self.learningRate = tf.placeholder("float")
        self.probability = tf.placeholder("float", [None, self.netPara.GetLabelSize()])
        #self.pred = self.multilayerLSTMNetForOneSentencePlaceholder(self.sequenceBatch, self.sourceTargetPlace)
        self.pred, self.testF = self.multilayerLSTMNetModern(self.sequenceBatch, self.sourceTargetPlace)
        self.calculatedProb = tf.nn.softmax(self.pred)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probability,logits=self.pred))
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate= self.learningRate).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learningRate).minimize(self.cost)
        self.init = tf.global_variables_initializer()
        

        # for translation
        self.translationPred = self.multilayerLSTMNetTranslationPredict(self.sequenceBatch, self.sourceTargetPlace)
        self.translationProb = tf.nn.softmax(self.translationPred)
        self.saver = tf.train.Saver()
        #initialize
        

        if (continue_pre == 1):
            print('read from trained file')
            self.saver.restore(self.sess, self.networkPathPrefix + 'lexiconModel')
        else:
            self.sess.run(self.init)



    def saveMatrixToFile(self):
        self.saver.save(self.sess, self.networkPathPrefix + 'lexiconModel')

    def multilayerLSTMNetModern(self, sequence_batch, _sourcetargetNum):

        _concatOutput = tf.zeros([0,self.readyToProcessDim])
        i0 = tf.constant(0)
        j0 = tf.constant(0)
        _output = tf.zeros([0,self.projOutDim])
        seqIndexSource, seqIndexTarget = tf.split(sequence_batch, _sourcetargetNum, 1)
        seqSource = tf.nn.embedding_lookup(self.weights_projection_source, seqIndexSource)
        seqTarget = tf.nn.embedding_lookup(self.weights_projection_target, seqIndexTarget)

        with tf.variable_scope("RNNLexicon"):

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

        with tf.variable_scope("RNNLexicon"):
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
        hiddenLayer1_ = tf.nn.tanh(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1_, self.weights_hidden2), self.biases_bHidden2)
        hiddenLayer2_ = tf.nn.tanh(hiddenLayer2)

        out = tf.add(tf.matmul(hiddenLayer2_, self.weights_out),self.biases_out)

        return out, hiddenLayer2

    '''
    def multilayerLSTMNetForOneSentencePlaceholder(self, sequence_batch, _sourceTargetNum):

        _sourceNum = _sourceTargetNum[0]
        _targetNum = _sourceTargetNum[1]
        _concatOutput = tf.zeros([0,400])

        cell = tf.contrib.rnn.BasicLSTMCell(200, forget_bias=0.0, state_is_tuple=True, reuse=None)
        #initial state
        zeroState  = cell.zero_state(self.batchSize, tf.float32)
        concatVector = tf.nn.embedding_lookup(self.weights_projection, sequence_batch)
        # here if you change batchSize from 1 to other value here maybe something wrong.
        _stateC = zeroState[0]
        _stateH = zeroState[1]
        i0 = tf.constant(0)
        j0 = tf.constant(0)
        _output = tf.zeros([0,200])
        

        # just to initialize the variables
        with tf.variable_scope("RNNLexicon"):
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

        with tf.variable_scope("RNNLexicon"):

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

        with tf.variable_scope("RNNLexicon"):

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

        
    def networkPrognose(self, _sequenceBatch, lexiconLabel, _sourceNum, _targetNum):

        self.output, fuck = self.sess.run([self.calculatedProb, self.testF],feed_dict={self.sequenceBatch : _sequenceBatch,
            self.sourceTargetPlace : [_sourceNum, _targetNum]})
        outProbability = []
        out = self.output
        print(out)
        for i in range(len(lexiconLabel)):
            outProbability.append(out[i][lexiconLabel[i]])
        return outProbability

    def trainingBatch(self, _sequenceBatch, batch_probability, _sourceNum, _targetNum, learningRate):
        _, c, fuck = self.sess.run([self.optimizer, self.cost, self.testF], feed_dict={self.sequenceBatch: _sequenceBatch,
                                self.probability: batch_probability,
                                self.sourceTargetPlace : [_sourceNum, _targetNum],
                                self.learningRate: learningRate})
        #print(fuck)
        return c

    def networkTranslationPrognose(self, _sequenceBatch, _sourceNum, _targetNum):

        self.output = self.sess.run([self.translationProb],feed_dict={self.sequenceBatch : _sequenceBatch,
            self.sourceTargetPlace : [_sourceNum, _targetNum]})
        outProbability = []
        out = self.output[0]
        return out