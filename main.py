#!/usr/bin/python
import data
import process

#initialize the data set
_data = data.ReadData()
targetClassSetSize = _data.getTargetClassSetSize()

#initialize the process
_process = process.ProcessTraditional(targetClassSetSize)

if _data.checkStatus() == 0:
	batchStatus = 0
	while True:
		batchStatus = _data.refreshNewBatch()
		if batchStatus == 1:
			break
		else:
			_process.processBatch(_data.getCurrentBatch())
			

	
else:
	print('stop the program')
	
