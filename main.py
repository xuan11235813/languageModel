#!/usr/bin/python
import data
import process


# specify the training epoch
epochMax = 6
epoch = 0

# configure the global settings
recordInterval = 100
measureInterval = 1
globalBatch = 0
measturePerplexity = 0

#initialize the data set
_data = data.ReadData()
_measureData = data.ReadData(1)

#initialize the process
_process = process.ProcessTraditional()
_process.processBatchWithBaumWelch(_data.getCurrentBatch())

if _data.checkStatus() == 0:
	batchStatus = 0
	while True:
		
		print('--------------------------')
		batchStatus = _data.refreshNewBatch()
		if batchStatus == 1:
			epoch += 1
			if epoch >= epochMax:
				break
			else:
				_data.refreshFilePosition()
				batchStatus = 0
				print('ready for epoch ' + repr(epoch))
		else:
			if epoch == 0:
				_process.processBatchWithBaumWelch(_data.getCurrentBatch())
			else:
				_process.processBatch(_data.getCurrentBatch())

		# for additional function including record the network and 
		# calculate the perplexity
		if globalBatch % recordInterval == recordInterval -1 :
			_process.recordNetwork()

		if globalBatch % measureInterval ==  measureInterval -1:
			_process.processPerplexity(_measureData.getCurrentBatch())
		globalBatch += 1

		print(len(_measureData.getCurrentBatch()))
		
else:
	print('stop the program')
	
