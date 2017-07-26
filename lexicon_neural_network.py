import tensorflow as tf
import para
import math as mt

class TraditionalLexiconNet:
    def __init__(self, targetClassSetSize):
        #parameter
        self.weights = {}
        self.biases = {}
        self.netPara = para.Para.LexiconNeuralNetwork()
        self.classSetSize = targetClassSetSize


        #network
        self.weightsInnerClass = []
        self.biasesInnerClass = []
        self.placeholderInnerClass = []

        self.weights['projection'] = tf.Variable(tf.random_normal(self.netPara.GetProjectionLayer()))
        self.weights['hidden1'] = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer1st()))
        self.weights['hidden2'] = tf.Variable(tf.random_normal(self.netPara.GetHiddenLayer2nd()))
        self.weights['outClass'] = tf.Variable(tf.random_normal(self.netPara.GetClassLayer()))
        self.biases['bHidden1'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer1st()[1]]))
        self.biases['bHidden2'] = tf.Variable(tf.random_normal([self.netPara.GetHiddenLayer2nd()[1]]))
        self.biases['outClass'] = tf.Variable(tf.random_normal([self.netPara.GetClassLayer()[1]]))

        
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

        #create series of class set
        for i in targetClassSetSize:
            if i <= 1:
                subLayer = [self.netPara.GetClassLayer()[0],1]
                item = tf.Variable(tf.random_normal(subLayer))
                itemBias = tf.Variable(tf.random_normal([i]))
                self.weightsInnerClass.append(item)
                self.biasesInnerClass.append(itemBias)
                
            else:

                subLayer = [self.netPara.GetClassLayer()[0],200]
                item = tf.Variable(tf.random_normal(subLayer))
                itemBias = tf.Variable(tf.random_normal([200]))
                self.weightsInnerClass.append(item)
                self.biasesInnerClass.append(itemBias)
                

        #initialize
        self.sess.run(self.init)

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

        outClass = tf.add(tf.matmul(hiddenLayer2, self.weights['outClass']),self.biases['outClass'])
        return outClass, hiddenLayer2

    def networkPrognose(self, sourceTarget, classAndClassIndex):
        self.output, middleOutput = self.sess.run([self.calculatedProb, self.middle],feed_dict={self.sequence : sourceTarget})
        outProbability = []
        sequenceIndex = []
        matrix = []
        biasMatrix = []
        innerIndexVector = []

        for i in range(len(classAndClassIndex)):
            classIndex = classAndClassIndex[i][0]
            innerIndex = classAndClassIndex[i][1]
            outProbability.append(self.output[i][classIndex])
            if self.classSetSize[classIndex] > 1:
                probBase = self.output[i][classIndex]
                innerIndexVector.append(innerIndex)
                #sequenceIndex.append(i)
                #matrix.append(self.weightsInnerClass[classIndex])
                #biasMatrix.append(self.biasesInnerClass[classIndex])
        '''
        if len(sequenceIndex) > 0:
            generateMiddle = tf.gather(middleOutput, sequenceIndex)
            listNum = len(sequenceIndex)
            itemNum = len(middleOutput[0])
            matrix = tf.reshape(tf.concat(matrix, 0), [listNum, itemNum ,200])
            flatVector = tf.reshape(generateMiddle, [listNum, 1 ,itemNum])
            biasMatrix = tf.reshape(tf.concat(biasMatrix, 0), [listNum, 200]  )
            multiResult = tf.reshape(tf.matmul(flatVector, matrix), [listNum, 200])
            result = tf.add(multiResult, biasMatrix)
            innerProb = tf.nn.softmax(result)
            innerOutput = self.sess.run(innerProb)
            '''

        return outProbability

    def trainingBatch(self, batch_sequence, batch_probabilityClass):
        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.sequence: batch_sequence,
                                self.probabilityClass: batch_probabilityClass})
        print(c)



