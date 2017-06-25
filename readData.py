#!/usr/bin/python
import sys
import para
import os

class WordAndClass:
	wordClass = -1
	word = ''

	def __init__(self, _word, _wordClass):
		self.word = _word
		self.wordClass = _wordClass


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
lst = []
for line in file:
	item = []
	for word in line.split(" "):
		item.append(word)
	print(item)
	wordAndClass = WordAndClass(item[0].rstrip(),item[2].rstrip())
	lst.append(wordAndClass)
	if index <= 10:
		index += 1
	else:
		break

print(lst.index(WordAndClass('very',57)))