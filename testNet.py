import lexicon_neural_network as ll
import alignment_neural_network as al
import data
import samples
import forward_backward as fb 

_data = data.ReadData()

net = ll.TraditionalLexiconNet()
net2 = al.TraditionalAlignmentNet()
FB = fb.ForwardBackward()


a = [[39230, 31011, 4, 18691, 9438, 8, 3688, 17389], 
[39230, 31011, 4, 18691, 9438, 8, 3688, 17389],
[39230, 31011, 4, 18691, 9438, 8, 3688, 17389]]

b = [[804, 16],[1504, 16],[1504, 16]]



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

