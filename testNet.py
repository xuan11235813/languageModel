import lexicon_neural_network as ll
import alignment_neural_network as al
import data
import samples
import forward_backward as fb 

'''
# -----------------------traditional neural network unit test-----------------

_data = data.ReadData()

net = ll.TraditionalLexiconNet()
net2 = al.TraditionalAlignmentNet()
FB = fb.ForwardBackward()

p = samples.GenerateSamples()
test = _data.trainingSentence[1];
targetNum, sourceNum = test. getSentenceSize()

samples, labels = p.getSimpleLexiconSamples(test)
samples2, initial = p.getAlignmentSamples(test)



o = net.networkPrognose(samples,labels)
o1, _ = net2.networkPrognose(samples2,initial)


gamma1, gamm2 = FB.calculateForwardBackward(o,o1, targetNum, sourceNum)

print(gamma1)

print(gamm2)

'''

#------------------------lstm neural network unit test---------------------

_data = data.ReadData()

lexiconNet = ll.LSTMLexiconNet()
alignmentNet = al.LSTMAlignmentNet()
FB = fb.ForwardBackward()

p = samples.GenerateSamples()
test = _data.trainingSentence[11]
targetNum, sourceNum = test. getSentenceSize()

print(test._source)
print(test._target)
print(test._targetClass)

samples, labels = p.getLSTMLexiconSample(test)
samples2 = p.getLSTMAlignmentSample(test)
print('----------------lexicon samples and labels-------------')
print(samples)
print(labels)
print('-------------length of labels-------------------')
print(len(labels))

print('----------------alignment samples-------------')
print(samples2)

print('--------------network print-----------------')
outLexicon = lexiconNet.networkPrognose(samples, labels, sourceNum, targetNum)

pinrt(out)