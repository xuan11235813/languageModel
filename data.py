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
		self._targetClass = []
		self._innerClassIndex = []
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




class ReadData:
	def __init__(self, measure = 0):

		self.log = printLog.Log()

		# these are inner parameters
		self.sourceVocab = []
		self.targetVocab = []
		self.targetClass = []
		self.trainingSentence = []
		self.trainFile = file
		self.trainFileCurrentPosition  = 0
		self.targetWordClassSet = []
		self.IBMDic = {}

		self.sourceVocab.append('')
		self.targetVocab.append('')
		self.targetClass.append('0')

		# initialize the target word class set
		for i in range(2000):
			item = []
			self.targetWordClassSet.append(item)
		self.targetWordClassSet[0].append('')

		# for any unsafe manipulate
		self.alert = 0

		# read the config file
		parameters = para.Para()
		self.parameters = parameters
		self.bias = parameters.GetTargetSourceBias()
		
		# read source dictionary
		self.sourceVocabFilePath = os.path.join(os.path.dirname(__file__), parameters.GetSourceVocabFilePath())
		self.readSourceDictionary(self.sourceVocabFilePath)

		# read target dictionary
		self.targetVocabFilePath = os.path.join(os.path.dirname(__file__), parameters.GetTargetVocabFilePath())
		self.readTargetDictionaryClass(self.targetVocabFilePath)

		
		# 
		if measure == 0:

			self.IBM1DataFilePath = os.path.join(os.path.dirname(__file__), parameters.GetIBMFilePath())
			self.readIBM1Data(self.IBM1DataFilePath)
			self.trainingDataFilePath = os.path.join(os.path.dirname(__file__), parameters.GetTrainingDataFilePath())
			self.readTrainDataBatch(self.trainingDataFilePath, self.parameters.ContinueOrRestart())
		else:
			self.trainingDataFilePath = os.path.join(os.path.dirname(__file__), parameters.GetMeasureDataFilePath())
			self.readMeasureDataBatch(self.trainingDataFilePath)


		

		#print(self.sourceVocab.index(''))
		#print(self.targetVocab.index(''))
		#print(len(self.sourceVocab))
		#print(self.trainingSentence[0]._source)
		#print(self.trainingSentence[0]._target)
		#print(self.trainingSentence[0]._targetClass)
		#print(self.trainingSentence[0]._innerClassIndex)
		#print(len(self.IBMDic))

		if self.alert != 0 :
			print('insufficient input data file')
		else:
			print('vocabulary and first batch ready')

			
	def recordCurrentTrainPosition(self):
		self.log.writeRecordInformation(repr(self.trainFileCurrentPosition))

	def findSourceVocabIndex( self, word ):
		value = -1
		try:
			value = self.sourceVocab.index( word )
			if value == 0:
				value = -1
		except ValueError as err:
			value = -1
		return value
	def findTargetVocabIndexAndClass(self, word ):
		value = []
		try:
			value.append(self.targetVocab.index( word ))
			value.append(int(self.targetClass[self.targetVocab.index( word )]))
			if value[0] == 0:
				value[0] = -1
		except ValueError as err:
			value.append(-1)
		return value


	def readSourceDictionary( self, filePath ):
		try:
			file = open(filePath, "r")
			for line in file:
				self.sourceVocab.append(line.rstrip())
		except IOError as err:
			print("source vocabulary files do not exist")
			self.alert += 1

	def readTargetDictionaryClass( self, filePath ):
		try:
			file = open(filePath, "r")
			for line in file:
				item = []
				for word in line.split(" "):
					item.append(word)
				self.targetVocab.append(item[0].rstrip())
				self.targetWordClassSet[int(item[2].rstrip())].append(item[0].rstrip())
				self.targetClass.append(item[2].rstrip())
		except IOError as err:
			print("target vocabulary files do not exist")
			self.alert += 1

	def readTrainDataBatch(self, filePath, continue_pre = 0):
		_allSource = 0
		_allTarget = 0
		sourceAbnormal = 0
		targetAbnormal = 0

		if continue_pre != 0:
			pos = self.log.readRecordInformation()
			self.trainFileCurrentPosition = pos

		try:
			self.trainFile = open(filePath, "r")
			for index in range(128):
				line = self.trainFile.readline()
				item = []
				sentencePair = SentencePair()
				for sentence in line.split('#'):
					item.append(sentence)
				for word in item[0].rstrip().split(' '):
					returnValue = self.findSourceVocabIndex(word)
					_allSource += 1
					if returnValue != -1:
						sentencePair._source.append(returnValue)
					else:
						sourceAbnormal+= 1
				for word in item[1].rstrip().split(' '):
					returnValue = self.findTargetVocabIndexAndClass(word)
					_allTarget += 1
					if returnValue[0] != -1:
						innerClass = self.targetWordClassSet[int(returnValue[1])].index(word)
						sentencePair._innerClassIndex.append(innerClass)
						sentencePair._target.append(returnValue[0])
						sentencePair._targetClass.append(returnValue[1])

					else:
						targetAbnormal += 1

				prob = self.findIBM1Prob(sentencePair._source, sentencePair._target)
				sentencePair._IBM1Data = prob
				if sentencePair.checkSentence() == 0:
					self.trainingSentence.append(sentencePair)
			#print('target training data with noise: ' + str(targetAbnormal/float(_allTarget)))
			#print('source training data with noise: ' + str(sourceAbnormal/float(_allSource)))
			
			self.trainFileCurrentPosition = self.trainFile.tell()

		except IOError as err:
			print("training files do not exist")
			self.alert += 1

	def refreshNewBatch(self):
		_allSource = 0
		_allTarget = 0
		sourceAbnormal = 0
		targetAbnormal = 0
		batchReady = 0
		del self.trainingSentence[:]
		try:
			self.trainFile = open(self.trainingDataFilePath, "r")
			self.trainFile.seek(self.trainFileCurrentPosition, 0)

			for index in range(128):
				line = self.trainFile.readline()
				if line == "":
					batchReady = 1
					break
				item = []
				sentencePair = SentencePair()
				for sentence in line.split('#'):
					item.append(sentence)
				for word in item[0].rstrip().split(' '):
					returnValue = self.findSourceVocabIndex(word)
					_allSource += 1
					if returnValue != -1:
						sentencePair._source.append(returnValue)
					else:
						sourceAbnormal+= 1
				for word in item[1].rstrip().split(' '):
					returnValue = self.findTargetVocabIndexAndClass(word)
					_allTarget += 1
					if returnValue[0] != -1:
						innerClass = self.targetWordClassSet[int(returnValue[1])].index(word)
						sentencePair._innerClassIndex.append(innerClass)
						sentencePair._target.append(returnValue[0])
						sentencePair._targetClass.append(returnValue[1])
						
					else:
						targetAbnormal += 1
				prob = self.findIBM1Prob(sentencePair._source, sentencePair._target)
				sentencePair._IBM1Data = prob
				if sentencePair.checkSentence() == 0:
					self.trainingSentence.append(sentencePair)
			
			self.trainFileCurrentPosition = self.trainFile.tell()
			if len(self.trainingSentence) ==  128:
				print('read a batch')
			else:
				print('read an incomplete batch')
		except IOError as err:
			print("training files do not exist")
			self.alert += 1
		return batchReady
	def readMeasureDataBatch(self, filePath):
		_allSource = 0
		_allTarget = 0
		sourceAbnormal = 0
		targetAbnormal = 0
		try:
			self.trainFile = open(filePath, "r")
			while True:
				line = self.trainFile.readline()
				if line  == '' or len(self.trainingSentence) == self.parameters.GetTestBatchSize():
					break;
				item = []
				sentencePair = SentencePair()
				for sentence in line.split('#'):
					item.append(sentence)
				for word in item[0].rstrip().split(' '):
					returnValue = self.findSourceVocabIndex(word)
					_allSource += 1
					if returnValue != -1:
						sentencePair._source.append(returnValue)
					else:
						sourceAbnormal+= 1
				for word in item[1].rstrip().split(' '):
					returnValue = self.findTargetVocabIndexAndClass(word)
					_allTarget += 1
					if returnValue[0] != -1:
						innerClass = self.targetWordClassSet[int(returnValue[1])].index(word)
						sentencePair._innerClassIndex.append(innerClass)
						sentencePair._target.append(returnValue[0])
						sentencePair._targetClass.append(returnValue[1])

					else:
						targetAbnormal += 1

				if sentencePair.checkSentence() == 0:
					self.trainingSentence.append(sentencePair)
			#print('target training data with noise: ' + str(targetAbnormal/float(_allTarget)))
			#print('source training data with noise: ' + str(sourceAbnormal/float(_allSource)))
			
			self.trainFileCurrentPosition = self.trainFile.tell()

		except IOError as err:
			print("training files do not exist")
			self.alert += 1
			
	def getTargetClassSetSize(self):
		targetClassSet = []
		for i in self.targetWordClassSet:
			targetClassSet.append(len(i))
		return targetClassSet
		
	def getCurrentBatch(self):
		return self.trainingSentence

	def checkStatus(self):
		if self.alert == 0:
			return 0
		else:
			return 1


	def readIBM1Data(self, filePath):
		try:
			file = open(filePath, "r")
			for line in file:
				item = []
				for word in line.split(" "):
					item.append(word)
				itemDic = {}
				itemDic[item[1].rstrip()] = float(item[2].rstrip())
				if item[0].rstrip() in self.IBMDic:
					self.IBMDic[item[0].rstrip()][item[1].rstrip()] = float(item[2].rstrip())
				else:
					self.IBMDic[item[0].rstrip()] = itemDic


		except IOError as err:
			print("target vocabulary files do not exist")
			self.alert += 1
		
	def findIBM1Prob(self, source, target):
		prob = []

		for i in target:
			for j in source:
				sourceWord = self.sourceVocab[j]
				targetWord = self.targetVocab[i]
				if sourceWord in self.IBMDic:
					if targetWord in self.IBMDic[sourceWord]:
						probItem  = self.IBMDic[sourceWord][targetWord]
					else:
						probItem = 0
				else:
					print('vocabulary wrong')
				
				prob.append(probItem)

		return prob

	def refreshFilePosition(self):
		self.trainFileCurrentPosition  = 0
