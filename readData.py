#!/usr/bin/python
import sys
import para
import os
class SentencePair:

	_source = []
	_target = []
	_targetClass = [] 



class ReadData:

	# these parameters should be initialized at one place
	sourceVocabFilePath = ""
	targetVocabFilePath = ""
	trainingDataFilePath = ""

	# these are inner parameters
	sourceVocab = []
	targetVocab = []
	targetClass = []
	trainingSentence = []

	allert = 0


	def __init__(self):

		# read the config file
		parameters = para.Para()

		# read source dictionary
		self.sourceVocabFilePath = os.path.join(os.path.dirname(__file__), parameters.GetSourceVocabFilePath())
		self.readSourceDictionary(self.sourceVocabFilePath)

		# read target dictionary
		self.targetVocabFilePath = os.path.join(os.path.dirname(__file__), parameters.GetTargetVocabFilePath())
		self.readTargetDictionaryClass(self.targetVocabFilePath)

		self.trainingDataFilePath = os.path.join(os.path.dirname(__file__), parameters.GetTrainingDataFilePath())
		self.readTrainData(self.trainingDataFilePath)


		if self.allert != 0 :
			print('insufficient input data file')
		else:
			print('data ready')

	def readSourceDictionary( self, filePath ):
		try:
			file = open(filePath, "r")
			for line in file:
				self.sourceVocab.append(line.rstrip())
		except IOError as err:
			print("source vocabulary files do not exist")
			self.allert += 1

	def readTargetDictionaryClass( self, filePath ):
		try:
			file = open(filePath, "r")
			for line in file:
				item = []
				for word in line.split(" "):
					item.append(word)
				self.targetVocab.append(item[0].rstrip())
				self.targetClass.append(item[2].rstrip())
		except IOError as err:
			print("target vocabulary files do not exist")
			self.allert += 1

	def readTrainData(self, filePath):
		try:
			file = open(filePath, "r")
			for line in file:
				item = []
				sentencePair = SentencePair()
				for sentence in line.split('#'):
					item.append(sentence)
				for word in item[0].rstrip().split(' '):
					sentencePair._source.append(self.sourceVocab.index(word))
				for word in item[1].rstrip().split(' '):
					sentencePair._target.append(self.targetVocab.index(word))
					sentencePair._targetClass.append(self.targetClass[self.targetVocab.index(word)])
				print(sentencePair._source)
				print(item[0])
				self.trainingSentence.append(sentencePair)

		except IOError as err:
			print("training files do not exist")
			self.allert += 1
		except ValueError as err:
			print("abnormal input words")




a = ReadData()


