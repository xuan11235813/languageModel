#!/usr/bin/python
import dataInput as data
import process
import printLog
import para

parameter = para.Para()


_data = data.ReadData(parameter.ReadTrainingFile())

_log = printLog.Log()

print(_data.getCurrentBatch()[0]._source)


#initialize the process
#_process = process.ProcessTraditional()

_process = process.ProcessLSTM()

_process.processPerplexity(_data.getCurrentBatch())


_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())
_process.processBatchWithBaumWelch(_data.getCurrentBatch())
_process.processPerplexity(_data.getCurrentBatch())

print("hahahahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")

for i in range(500):
	_process.processBatchWithBaumWelch(_data.getCurrentBatch())
	_process.processPerplexity(_data.getCurrentBatch())
_process.recordNetwork()

'''

_process_ = process.ProcessLSTM()

#_process_.processBatch(_data.getCurrentBatch())

#_process_.processPerplexity(_data.getCurrentBatch())
_process_ .testOneSentence( _data.getCurrentBatch()[0] )

_process_ .testOneSentence( _data.getCurrentBatch()[1] )
'''