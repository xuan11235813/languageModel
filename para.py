class Para:

	#for Data
	sourceVocabFilePath = "data/sourceVocab"
	targetVocabFilePath = "data/engClass"
	trainingDataFilePath = "data/dev"


	def GetSourceVocabFilePath(self):
		return self.sourceVocabFilePath
	def GetTargetVocabFilePath(self):
		return self.targetVocabFilePath
	def GetTrainingDataFilePath(self):
		return self.trainingDataFilePath

	#for lexicon neural network
	