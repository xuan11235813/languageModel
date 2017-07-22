import lexicon_neural_network as ll
import data
import samples

_data = data.ReadData()
targetClassSetSize = _data.getTargetClassSetSize()

net = ll.TraditionalLexiconNet(targetClassSetSize)


a = [[39230, 31011, 4, 18691, 9438, 8, 3688, 17389], 
[39230, 31011, 4, 18691, 9438, 8, 3688, 17389],
[39230, 31011, 4, 18691, 9438, 8, 3688, 17389]]

b = [[804, 16],[1504, 16],[1504, 16]]



p = samples.GenerateSamples()
test = _data.trainingSentence[1];
print(test._source)
print(test._target)
print(test._targetClass)
print(test._innerClassIndex)
samples, labels = p.getLexiconSamples(test)
samples2 = p.getAlignmentSamples(test)

print(samples2)

o = net.networkPrognose(samples,labels)
print(o)