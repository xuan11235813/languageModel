import dataInput
from subprocess import call
data = dataInput.ReadData()
dic = dataInput.ReadDic()
batch = data.getCurrentBatch()
data.refreshNewBatch()
data.refreshNewBatch()
batch = data.getCurrentBatch()
sentence = batch[0]
print(sentence.getIBMLexiconInitialDatax`())
print(sentence._source)

for i  in sentence._source:
	print(dic.findSourceWord(i))
print(sentence._target)
for i in sentence._target:
	print(dic.findTargetWord(i))

data.recordCurrentTrainPosition()

data_ = dataInput.ReadData()
batch = data_.getCurrentBatch()
sentence = batch[0]

print(sentence._source)

for i  in sentence._source:
	print(dic.findSourceWord(i))
print(sentence._target)
for i in sentence._target:
	print(dic.findTargetWord(i))

call(['rm', 'record.config.log'])