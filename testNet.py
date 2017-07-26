import lexicon_neural_network as ll
import alignment_neural_network as al
import data
import samples
import forward_backward as fb 

_data = data.ReadData()
targetClassSetSize = _data.getTargetClassSetSize()

net = ll.TraditionalLexiconNet(targetClassSetSize)
net2 = al.TraditionalAlignmentNet()
FB = fb.ForwardBackward()


a = [[39230, 31011, 4, 18691, 9438, 8, 3688, 17389], 
[39230, 31011, 4, 18691, 9438, 8, 3688, 17389],
[39230, 31011, 4, 18691, 9438, 8, 3688, 17389]]

b = [[804, 16],[1504, 16],[1504, 16]]



p = samples.GenerateSamples()
test = _data.trainingSentence[1];
targetNum, sourceNum = test. getSentenceSize()
print('fuck2')
samples, labels = p.getLexiconSamples(test)
samples2, _ = p.getAlignmentSamples(test)

print('fuck3')

o = net.networkPrognose(samples,labels)
o1 = net2.networkPrognose(samples2)

print(o)
print(o1)
gamma1, gamm2 = FB.calculateForwardBackward(o,o1, targetNum, sourceNum)
print('haha')
print(gamma1)
print('hhaha')
print(gamm2)

