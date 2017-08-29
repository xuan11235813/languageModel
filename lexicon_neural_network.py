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
            self.weights['out'] = tf.Variable(tf.random_normal(self.netPara.GetClassLayer()))
            self.biases['bHidden1'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer1st()[1]]))
            self.biases['bHidden2'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer2nd()[1]]))
            self.biases['out'] = tf.Variable(tf.random_normal([self.netPara.GetClassLayer()[1]]))
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
        self.probabilityClass = tf.placeholder("float", [None, self.netPara.GetClassLabelSize()])
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
        hiddenLayer1 = tf.nn.sigmoid(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, self.weights['hidden2']), self.biases['bHidden2'])
        hiddenLayer2 = tf.nn.sigmoid(hiddenLayer2)

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



