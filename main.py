#!/usr/bin/python
import Data as data

_data = data.ReadData()

if _data.checkStatus() == 0:
	batchStatus = 0
	while True:
		batchStatus = _data.refreshNewBatch()
		if batchStatus == 1:
			break

	
else:
	print('stop the program')
	
