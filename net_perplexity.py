#!/usr/bin/python
import dataInput as data
import process
import printLog
import para

parameter = para.Para()


_data = data.ReadData(parameter.ReadTrainingFile())

_log = printLog.Log()

#initialize the process
#_process = process.ProcessTraditional()

_process = process.ProcessLSTM()

_process.processPerplexity(_data.getCurrentBatch())

_process.processBatchWithBaumWelch(_data.getCurrentBatch())

for i in range(10):
	_process.processBatch(_data.getCurrentBatch())
	_process.processPerplexity(_data.getCurrentBatch())
_process.recordNetwork()

for i in range(10):
	_process.processBatch(_data.getCurrentBatch())
	_process.processPerplexity(_data.getCurrentBatch())
_process.recordNetwork()

	
_process_ = process.ProcessLSTM()

_process_.processPerplexity(_data.getCurrentBatch())