#!/usr/bin/python
import sys
import para
import os
import numpy as np
import printLog


'''
This class stores the target sentence index and souce sentence
index.
Besides the IBM data will be stored here.

'''

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

'''
This class store the dictionary generated from training file
After dcitionary generation you can easily find word index or find word by index
'''
class ReadDic:
	def __init__(self):
		parameter = para.Para()
		self.sourceDic = {}
		self.targetDic = {}
		self.sourceIndexDic = {}
		self.targetIndexDic = {}
		self.alert = 0

		self.sourceTrainingFilePath = os.path.join(os.path.dirname(__file__),parameter.GetSourceTrainingFilePath())
		self.targetTrainingFilePath = os.path.join(os.path.dirname(__file__),parameter.GetTargetTrainingFilePath())

		try:
			self.sourceTrainingFile = open(self.sourceTrainingFilePath, 'r')
			self.targetTrainingFile = open(self.targetTrainingFilePath, 'r')

			sourceIndex = 1
			for line in self.sourceTrainingFile:
				for word in line.split(" "):
					item  = word.rstrip()
					if item not in self.sourceDic:
						self.sourceDic[item] = sourceIndex
						self.sourceIndexDic[sourceIndex] = item
						sourceIndex += 1

			targetIndex = 1
			for line in self.targetTrainingFile:
				for word in line.split(" "):
					item = word.rstrip()
					if item not in self.targetDic:
						self.targetDic[item] = targetIndex
						self.targetIndexDic[targetIndex] = item
						targetIndex += 1		

			self.targetTrainingFile.close()
			self.sourceTrainingFile.close()

		except IOError as err:
			print("training files do not exist")
			self.alert += 1
		

	def findSourceIndex( self, word ):
		try:
			index  = self.sourceDic[word]
		except KeyError as err:
			index = -1
		return index
	def findTargetIndex( self, word ):
		try:
			index  = self.targetDic[word]
		except KeyError as err:
			index = -1
		return index
	def findSourceWord( self, index ):
		return self.sourceIndexDic[index]
	def findTargetWord( self, index ):
		return self.targetIndexDic[index]
	def isNormal(self):
		return self.alert



class ReadIBM:
	def __init__(self):
		parameter = para.Para()
		self.IBMDic = {}
		self.IBMDataFilePath1 = os.path.join(os.path.dirname(__file__),parameter.GetIBMDataFile1())
		self.IBMDataFilePath2 = os.path.join(os.path.dirname(__file__),parameter.GetIBMDataFile2())
		self.alert = 0

		try:
			IBMFile1 = open(self.IBMDataFilePath1, "r")
			IBMFile2 = open(self.IBMDataFilePath2, "r")
			for line in IBMFile1:
				item = []
				for word in line.split(" "):
					item.append(word)
				itemDic = {}
				itemDic[item[1].rstrip()] = float(item[2].rstrip())
				if item[0].rstrip() in self.IBMDic:
					self.IBMDic[item[0].rstrip()][item[1].rstrip()] = float(item[2].rstrip())
				else:
					self.IBMDic[item[0].rstrip()] = itemDic

			for line in IBMFile2:
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
			print("IBM files do not exist or incomplete")
			self.alert += 1
	def findProb( self, sourceWord, targetWord):
		if sourceWord in self.IBMDic:
			if targetWord in self.IBMDic[sourceWord]:
				prob = self.IBMDic[sourceWord][targetWord]
			else:
				prob = 0
		else:
			prob = 0
		return prob
	def findIBMProb(self, source, target):
		prob = []
		for targetWord in target:
			for sourceWord in source:
				probItem = self.findProb(sourceWord, targetWord)
				prob.append(probItem)
	def isNormal(self):
		return self.alert


class ReadData:
	def __init__(self, measure = 0):
		self.log = printLog.Log()
		self.trainingSentence = []
		self.parameter = para.Para()
		self.sourceTrainingFilePath = os.path.join(os.path.dirname(__file__), self.parameter.GetSourceTrainingFilePath())
		self.targetTrainingFilePath = os.path.join(os.path.dirname(__file__), self.parameter.GetTargetTrainingFilePath())
		self.sourceMeasureFilePath = os.path.join(os.path.dirname(__file__), self.parameter.GetSourceMeasureFilePath())
		self.targetMeasureFilePath = os.path.join(os.path.dirname(__file__), self.parameter.GetTargetMeasureFilePath())
		
		self.sourceTrainFileCurrentPosition = 0
		self.targetTrainFileCurrentPosition = 0

		self.alert = 0	
		self.vocabDic = ReadDic()
		self.alert += self.vocabDic.isNormal()

		if measure == 0:
			self.IBMWordDic = ReadIBM()
			self.alert += self.IBMWordDic.isNormal()
			self.readTrainDataBatch(self.parameter.ContinueOrRestart())
		else:
			self.readMeasureDataBatch()

		if self.alert != 0 :
			print('insufficient input data file')
		else:
			print('vocabulary and first batch ready')
		


	def recordCurrentTrainPosition(self):
		self.log.saveParameter('sourcePosition', repr(self.sourceTrainFileCurrentPosition))
		self.log.saveParameter('targetPosition', repr(self.targetTrainFileCurrentPosition))
	
	def refreshFilePosition(self):
		self.sourceTrainFileCurrentPosition = 0
		self.targetTrainFileCurrentPosition = 0
	
	def readTrainDataBatch( self , continue_pre = 0 ):
		if continue_pre != 0:
			posSource = self.log.readParameter('sourcePosition')
			posTarget = self.log.readParameter('targetPosition')
			self.sourceTrainFileCurrentPosition = int(posSource)
			self.targetTrainFileCurrentPosition = int(posTarget)

		try:
			self.sourceTrainingFile = open(self.sourceTrainingFilePath, 'r')
			self.targetTrainingFile = open(self.targetTrainingFilePath, 'r')
			self.sourceTrainingFile.seek(self.sourceTrainFileCurrentPosition, 0)
			self.targetTrainingFile.seek(self.targetTrainFileCurrentPosition, 0)

			for index in range(128):
				sourceLine = self.sourceTrainingFile.readline()
				targetLine = self.targetTrainingFile.readline()
				sourceWordList = []
				targetWordList = []
				sentencePair = SentencePair()
				for sourceWord in sourceLine.rstrip().split(' '):
					sourceIndex = self.vocabDic.findSourceIndex( sourceWord )
					if sourceIndex != -1:
						sentencePair._source.append(sourceIndex)
						sourceWordList.append(sourceWord)
				for targetWord in targetLine.rstrip().split(' '):
					targetIndex = self.vocabDic.findTargetIndex( targetWord )
					if targetIndex != -1:
						sentencePair._target.append(targetIndex)
						targetWordList.append(targetWord)
				prob = self.IBMWordDic.findIBMProb(sourceWordList, targetWordList)
				sentencePair._IBM1Data = prob
				if sentencePair.checkSentence() == 0:
					self.trainingSentence.append(sentencePair)
			self.sourceTrainFileCurrentPosition = self.sourceTrainingFile.tell()
			self.targetTrainFileCurrentPosition = self.targetTrainingFile.tell()

			self.sourceTrainingFile.close()
			self.targetTrainingFile.close()

		except IOError as err:
			print('training files do not exist')
			self.alert += 1

	def refreshNewBatch(self):
		batchReady = 0
		del self.trainingSentence[:]
		try:
			self.sourceTrainingFile = open(self.sourceTrainingFilePath, 'r')
			self.targetTrainingFile = open(self.targetTrainingFilePath, 'r')
			self.sourceTrainingFile.seek(self.sourceTrainFileCurrentPosition, 0)
			self.targetTrainingFile.seek(self.targetTrainFileCurrentPosition, 0)

			for index in range(128):
				sourceLine = self.sourceTrainingFile.readline()
				targetLine = self.targetTrainingFile.readline()
				if sourceLine == "" or targetLine == "":
					batchReady = 1
					break

				sourceWordList = []
				targetWordList = []
				sentencePair = SentencePair()
				for sourceWord in sourceLine.rstrip().split(' '):
					sourceIndex = self.vocabDic.findSourceIndex( sourceWord )
					if sourceIndex != -1:
						sentencePair._source.append(sourceIndex)
						sourceWordList.append(sourceWord)
				for targetWord in targetLine.rstrip().split(' '):
					targetIndex = self.vocabDic.findTargetIndex( targetWord )
					if targetIndex != -1:
						sentencePair._target.append(targetIndex)
						targetWordList.append(targetWord)
				prob = self.IBMWordDic.findIBMProb(sourceWordList, targetWordList)
				sentencePair._IBM1Data = prob
				if sentencePair.checkSentence() == 0:
					self.trainingSentence.append(sentencePair)
			self.sourceTrainFileCurrentPosition = self.sourceTrainingFile.tell()
			self.targetTrainFileCurrentPosition = self.targetTrainingFile.tell()
			if len(self.trainingSentence) == 128:
				print('read a batch')
			else:
				print('read an incomplete batch')
			self.sourceTrainingFile.close()
			self.targetTrainingFile.close()

		except IOError as err:
			print('training files do not exist')
			self.alert += 1
		return batchReady
	def readMeasureDataBatch(self):
		try:
			self.sourceMeasureFile = open(self.sourceMeasureFilePath, 'r')
			self.targetMeasureFile = open(self.targetMeasureFilePath, 'r')


			while True:
				sourceLine = self.sourceTrainingFile.readline()
				targetLine = self.targetTrainingFile.readline()
				if sourceLine =='' or targetLine == '' or len(self.trainingSentence) == self.parameters.GetTestBatchSize():
					break
				sourceWordList = []
				targetWordList = []
				sentencePair = SentencePair()
				for sourceWord in sourceLine.rstrip().split(' '):
					sourceIndex = self.vocabDic.findSourceIndex( sourceWord )
					if sourceIndex != -1:
						sentencePair._source.append(sourceIndex)
						sourceWordList.append(sourceWord)
				for targetWord in targetLine.rstrip().split(' '):
					targetIndex = self.vocabDic.findTargetIndex( targetWord )
					if targetIndex != -1:
						sentencePair._target.append(targetIndex)
						targetWordList.append(targetWord)
				if sentencePair.checkSentence() == 0:
					self.trainingSentence.append(sentencePair)
			self.sourceTrainFileCurrentPosition = self.sourceTrainingFile.tell()
			self.targetTrainFileCurrentPosition = self.targetTrainingFile.tell()

			self.sourceTrainingFile.close()
			self.targetTrainingFile.close()

		except IOError as err:
			print('training files do not exist')
			self.alert += 1
	
	def getCurrentBatch(self):
		return self.trainingSentence

	def checkStatus(self):
		if self.alert == 0:
			return 0
		else:
			return 1