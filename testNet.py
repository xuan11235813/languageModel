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

samplesLexicon, labelsLexicon = p.getLSTMLexiconSample(test)
samplesAlignment = p.getLSTMAlignmentSample(test)
print('----------------lexicon samples and labels-------------')
print(samplesLexicon)
print(labelsLexicon)
print('-------------length of labels-------------------')
print(len(labelsLexicon))

print('----------------alignment samples-------------')
print(samplesAlignment)

print('--------------network print-----------------')
outputLexicon = lexiconNet.networkPrognose([samplesLexicon], labelsLexicon, sourceNum, targetNum)

print(outputLexicon)
print(len(outputLexicon))

outputAlignment, outputAlignmentInitial = alignmentNet.networkPrognose([samplesAlignment], sourceNum, targetNum)

gamma, alignmentGamma = FB.calculateForwardBackward(outputLexicon,outputAlignment, targetNum, sourceNum)

print(gamma)

print(alignmentGamma)

lexiconLabel, alignmentLabel, alignmentLabelInitial  = p.getLabelFromGamma(alignmentGamma, gamma, test)
print('--------------lexiconLabel-------------')
print(lexiconLabel)

print('--------------alignmentLabel-------------')
print(alignmentLabel)

print('---------------alignment initial-------------------')
print(alignmentLabelInitial)

# train the network
costLexicon = lexiconNet.trainingBatch([samplesLexicon], lexiconLabel, sourceNum, targetNum)
			
# use data training alignment neural network
costAlignment = alignmentNet.trainingBatchWithInitial([samplesAlignment], alignmentLabel, alignmentLabelInitial, sourceNum, targetNum)