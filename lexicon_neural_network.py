import tensorflow as tf
import para

class LexiconNet:
    def __init__(self):
        #parameter
        self.weights = {}
        self.biases = {}
        self.netPara = para.Para.LexiconNeuralNetwork()

        #network
        self.weights['projection'] = tf.Variable(tf.random_normal(netPara.GetProjectionLayer()))
        self.weights['hidden1'] = tf.Variable(tf.random_normal(netPara.GetHiddenLayer1st()))
        self.weights['hidden2'] = tf.Variable(tf.random_normal(netPara.GetHiddenLayer2nd()))
        self.weights['outClass'] = tf.variable(tf.random_normal(netPara.GetClassLayer))
        self.biases['bHidden1'] = tf.Variable(tf.random_normal(netPara.GetHiddenLayer1st()[1]))
        self.biases['bHidden2'] = tf.Variable(tf.random_normal(netPara.GetHiddenLayer2nd()[1]))
        self.biases['outClass'] = tf.Variable(tf.random_normal(netPara.GetClassLayer[1]))

        #placeholder
        self.sequence = tf.placeholder(tf.int32, [None, netPara.GetInputWordNum()])
        self.probabilityClass = tf.placeholder("float", [None, netPara.GetClassLabelSize()])
        self.pred = self.multilayer_perceptron(sequence)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-1 * cost)
        self.init = tf.global_variables_initializer();
        sess = tf.Session()
        #initialize
        sess.run(init)

    def multilayer_perceptron(sourceTarget):
        concatVector = tf.Variable([], dtype = tf.float32)
        for word in sourceTarget:
            concatVector = tf.concat([concatVector,tf.gather(weights['projection'],word)],0)

        hiddenLayer1 = tf.add(tf.matmul(concatVector, weights['hidden1']), biases['bHidden1'])
        hiddenLayer1 = tf.nn.relu(hiddenLayer1)

        hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, weights['hidden2']), biases['bHidden2'])
        hiddenLayer2 = tf.nn.relu(hiddenLayer2)

        outClass = tf.add(tf.matmul(hiddenLayer2, weights['outClass']),biases['outClass'])

    def trainingBatch(batch_sequence, batch_probabilityClass):
        _, c = sess.run([optimizer, cost], feed_dict={self.sequence: batch_sequence,
                                self.probabilityClass: batch_probabilityClass})
        print(c)

    


