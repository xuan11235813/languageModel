#!/usr/bin/python
import data
import process


_data = data.ReadData()


#initialize the process
#_process = process.ProcessTraditional()
_process = process.ProcessLSTM()
_process.processUnitLexiconTest(_data.getCurrentBatch())

