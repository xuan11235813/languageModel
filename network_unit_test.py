#!/usr/bin/python
import data
import process


_data = data.ReadData()
targetClassSetSize = _data.getTargetClassSetSize()

#initialize the process
_process = process.ProcessTraditional(targetClassSetSize)

_process.processUnitLexiconTest(_data.getCurrentBatch())

