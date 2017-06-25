#!/usr/bin/python
import sys
import para
import os



class ReadData:

	# these parameters should be initialized at one place
	sourceVocabFilePath = ""
	targetVocabFilePath = ""

	# these are inner parameters
	sourceVocab = []
	targetVocab = []
	targetClass = []

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

	def readSourceDictionary( self, filePath ):
		try:
			file = open(filePath, "r")
			for line in file:
				self.sourceVocab.append(line.rstrip())
		except IOError as err:
			print("source vocabulary files not found")
			allert += 1

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
			print("source vocabulary files not found")
			allert += 1





a = ReadData()
