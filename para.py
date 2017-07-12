import tensorflow as tf
class Para:
	def __init__(self):
		#for source data file
		self.sourceVocabFilePath = "data/sourceVocab"
		self.targetVocabFilePath = "data/engClass"
		self.trainingDataFilePath = "data/dev"

		#training property
		self.batchSize = 128
	
	def GetSourceVocabFilePath(self):
		return self.sourceVocabFilePath
	def GetTargetVocabFilePath(self):
		return self.targetVocabFilePath
	def GetTrainingDataFilePath(self):
		return self.trainingDataFilePath


	


	#for lexicon neural network
	class LexiconNeuralNetwork:

		def __init__(self):
			self.lexiconSourceWindowSize = 5
			self.lexiconTargetWindowSize = 3
			self.projectionLayerInputDim = 40680
			self.projectionLayerOutputDim = 200
			self.hiddenLayer1stOutput = 1000
			self.hiddenLayer2ndOutput = 500
			self.outputLayerClassOutput = 2000
			self.outputLayerWordOutput = 40680

		def GetLexiconSourceWindowSize(self):			       	
			return self.lexiconSourceWindowSize
		def GetLexiconTargetWindowSize(self):
			return self.lexiconTargetWindowSize
		def GetInputWordNum(self):
			return self.lexiconSourceWindowSize + self.lexiconTargetWindowSize
		def GetProjectionLayer(self):
			return [self.projectionLayerInputDim, self.projectionLayerOutputDim]
		def GetHiddenLayer1st(self):
			inputNum = (self.lexiconTargetWindowSize + self.lexiconSourceWindowSize) * self.projectionLayerOutputDim
			return [inputNum, self.hiddenLayer1stOutput]
		def GetHiddenLayer2nd(self):
			return[self.hiddenLayer1stOutput, self.hiddenLayer2ndOutput]
		def GetClassLayer(self):
			return [hiddenLayer2ndOutput, outputLayerClassOutput]
		def GetWordOutputLayer(self):
			return [hiddenLayer2ndOutput, outputLayerWordOutput]
		def GetClassLabelSize(self):
			return outputLayerClassOutput