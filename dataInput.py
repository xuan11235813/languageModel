#!/usr/bin/python
import sys
import para
import os
import numpy as np
import printLog


class SentencePair:

	def __init__(self):
		self._source = []
		self._target = []
		self._IBM1Data = []


	def getSentenceSize(self):
		sourceNum = len(self._source)
		targetNum = len(self._target)
		return targetNum, sourceNum

	def getIBMLexiconInitialData(self):
		return self._IBM1Data

	def checkSentence(self):
		flag = 0
		if len(self._source) < 1:
			flag = 1
		if len(self._target) < 2:
			flag = 1
		return flag

class ReadDic:
	def __init__():
		

class ReadData:
	def __init__(self, measureSource = 0, measureTarget = 0):
		self.log = printLog.Log()
