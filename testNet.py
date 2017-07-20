import lexicon_neural_network as ll
import data 

_data = data.ReadData()
targetClassSetSize = _data.getTargetClassSetSize()

net = ll.TraditionalLexiconNet(targetClassSetSize)


a = [[39230, 31011, 4, 18691, 9438, 8, 3688, 17389], 
[39230, 31011, 4, 18691, 9438, 8, 3688, 17389],
[39230, 31011, 4, 18691, 9438, 8, 3688, 17389]]

b = [[1504, 16],[1504, 16],[1504, 16]]

o = net.networkPrognose(a,b)

print(o)