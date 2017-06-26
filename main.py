#!/usr/bin/python
import Data as data

_data = data.ReadData()

if _data.checkStatus() == 0:
	print('add something here')
else:
	print('stop the program')
	
