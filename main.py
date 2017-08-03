#!/usr/bin/python
import data
import process

# specify the training epoch
epochMax = 6
epoch = 0

#initialize the data set
_data = data.ReadData()
targetClassSetSize = _data.getTargetClassSetSize()

#initialize the process
_process = process.ProcessTraditional(targetClassSetSize)
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
			
		
else:
	print('stop the program')
	
