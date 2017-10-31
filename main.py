#!/usr/bin/python
import dataInput as data
import process
import printLog
import para

parameter = para.Para()

# specify the training epoch
epochMax = 6
epoch = 1

# configure the global settings
recordInterval = 20
measureInterval = 30
globalBatch = 0
measturePerplexity = 0

#initialize the data set
# ReadData() without input or with input parameter 0 means read the training file
# and 1 for reading from test file(measure data file)
_data = data.ReadData(parameter.ReadTrainingFile())
_measureData = data.ReadData(parameter.ReadTestFile())
_log = printLog.Log()

#initialize the process
#_process = process.ProcessTraditional()
_process = process.ProcessLSTM()
_process.processBatchWithBaumWelch(_data.getCurrentBatch())

if _data.checkStatus() == 0:
	batchStatus = 0
	while True:
		
		_log.writeSequence('epoch ' + repr(epoch) + ' batch: '+ repr(globalBatch))
		_log.writeSequence('---------------------------------------')
		batchStatus = _data.refreshNewBatch()
		if batchStatus == 1:
			epoch += 1
			if epoch >= epochMax:
				break
			else:
				_data.refreshFilePosition()
				batchStatus = 0

				_log.writeSequence('ready for epoch ' + repr(epoch))
		else:
			if epoch == 0:
				_process.processBatchWithBaumWelch(_data.getCurrentBatch())
			else:
				_process.processBatch(_data.getCurrentBatch())

		# for additional function including record the network and 
		# calculate the perplexity
		if globalBatch % recordInterval == recordInterval -1 :
			_process.recordNetwork()
			_data.recordCurrentTrainPosition()

		if globalBatch % measureInterval ==  measureInterval -1:
			_process.processPerplexity(_measureData.getCurrentBatch())
		globalBatch += 1
		
else:
	_log.writeSequence('stop the program')
	
