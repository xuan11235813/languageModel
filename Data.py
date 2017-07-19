#!/usr/bin/python
import sys
import para
import os

class SentencePair:

	def __init__(self):
		self._source = []
		self._target = []
		self._targetClass = []




class ReadData:
	def __init__(self):
		# these are inner parameters
		self.sourceVocab = []
		self.targetVocab = []
		self.targetClass = []
		self.trainingSentence = []
		self.trainFile = file
		self.trainFileCurrentPosition  = 0
		self.targetWordClassSet = []

		# initialize the target word class set
		for i in range(2000):
			item = []
			self.targetWordClassSet.append(item)

		# for any unsafe manipulate
		self.alert = 0

		# read the config file
		parameters = para.Para()

		# read source dictionary
		self.sourceVocabFilePath = os.path.join(os.path.dirname(__file__), parameters.GetSourceVocabFilePath())
		self.readSourceDictionary(self.sourceVocabFilePath)

		# read target dictionary
		self.targetVocabFilePath = os.path.join(os.path.dirname(__file__), parameters.GetTargetVocabFilePath())
		self.readTargetDictionaryClass(self.targetVocabFilePath)

		self.trainingDataFilePath = os.path.join(os.path.dirname(__file__), parameters.GetTrainingDataFilePath())
		self.readTrainDataBatch(self.trainingDataFilePath)


		for i in range(1999):
			print('*************')
			print(len(self.targetWordClassSet[i]))

		if self.alert != 0 :
			print('insufficient input data file')
		else:
			print('vocabulary and first batch ready')
	def findSourceVocabIndex( self, word ):
		value = -1
		try:
			value = self.sourceVocab.index( word )
		except ValueError as err:
			value = -1
		return value
	def findTargetVocabIndexAndClass(self, word ):
		value = []
		try:
			value.append(self.targetVocab.index( word ))
			value.append(int(self.targetClass[self.targetVocab.index( word )]))
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

	def readTrainDataBatch(self, filePath):
		_allSource = 0
		_allTarget = 0
		sourceAbnormal = 0
		targetAbnormal = 0
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
						sentencePair._target.append(returnValue[0])
						sentencePair._targetClass.append(returnValue[1])
						
					else:
						targetAbnormal += 1
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
						sentencePair._target.append(returnValue[0])
						sentencePair._targetClass.append(returnValue[1])
						
					else:
						targetAbnormal += 1
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

		

	def checkStatus(self):
		if self.alert == 0:
			return 0
		else:
			return 1

