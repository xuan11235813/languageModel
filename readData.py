#!/usr/bin/python
import sys
import para
import os



class readData:

	# these parameters should be initialized at one place
	sourceVocabFilePath = ""
	targetVocabFulePath = ""
	# these are inner parameters
	sourceVocab = []

	allert = 0


	def __init__(self):
		parameters = para.Para()

		self.sourceVocabFilePath = os.path.join(os.path.dirname(__file__), parameters.GetSourceVocabFilePath())
		self.readSourceDictionary(self.sourceVocabFilePath)
		print(self.sourceVocab.index("Quantifizierung"))


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
				self.sourceVocab.append(line.rstrip())
		except IOError as err:
			print("source vocabulary files not found")
			allert += 1




file = open('data/engClass', 'r')
index  = 0
lst = {}
for line in file:
	item = []
	for word in line.split(" "):
		item.append(word)
	lst[item[0].rstrip()] = item[1].rstrip()
	if index <= 10:
		index += 1
	else:
		break

