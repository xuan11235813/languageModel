
class Para:
	def __init__(self):

		# general configuration
		# if read network from files
		self.continue_pre = 1


		#for source data file
		self.sourceVocabFilePath = "data/sourceVocab"
		self.targetVocabFilePath = "data/engClass"
		self.trainingDataFilePath = "data/dev.train"
		self.IBMDataFilePath = 'data/prob'
		self.networkStoragePath = 'data/network/'
		self.measureDataFilePath = 'data/dev'

		#training property
		self.batchSize = 128
		self.testBatchSize = 40

	
	def GetSourceVocabFilePath(self):
		return self.sourceVocabFilePath
	def GetTargetVocabFilePath(self):
		return self.targetVocabFilePath
	def GetTrainingDataFilePath(self):
		return self.trainingDataFilePath
	def GetIBMFilePath(self):
		return self.IBMDataFilePath
	def ContinueOrRestart(self):
		return self.continue_pre
	def GetNetworkStoragePath(self):
		return self.networkStoragePath
	def GetMeasureDataFilePath(self):
		return self.measureDataFilePath
	def GetTestBatchSize(self):
		return self.testBatchSize
	def ReadTrainingFile(self):
		return 0
	def ReadTestFile(self):
		return 1


	#for lexicon neural network
	class LexiconNeuralNetwork:

		def __init__(self):
			self.lexiconSourceWindowSize = 5
			self.lexiconTargetWindowSize = 3
			self.projectionLayerInputDim = 40681
			self.projectionLayerOutputDim = 200
			self.hiddenLayer1stOutput = 1000
			self.hiddenLayer2ndOutput = 500
			self.outputLayerClassOutput = 2000
			self.outputLayerWordOutput = 40681
			self.learningRate = 0.01

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
			return [self.hiddenLayer2ndOutput, self.outputLayerClassOutput]
		def GetWordOutputLayer(self):
			return [self.hiddenLayer2ndOutput, self.outputLayerWordOutput]
		def GetClassLabelSize(self):
			return self.outputLayerClassOutput
		def GetLearningRate(self):
			return self.learningRate

	#for lexicon neural network
	class AlignmentNeuralNetwork:

		def __init__(self):
			self.alignmentSourceWindowSize = 5
			self.alignmentTargetWindowSize = 3
			self.projectionLayerInputDim = 40681
			self.projectionLayerOutputDim = 200
			self.hiddenLayer1stOutput = 1000
			self.hiddenLayer2ndOutput = 500
			self.outputLayerJumpOutput = 101
			self.learningRate = 0.01
			self.jumpLimited = 50

		def GetAlignmentSourceWindowSize(self):			       	
			return self.alignmentSourceWindowSize
		def GetAlignmentTargetWindowSize(self):
			return self.alignmentTargetWindowSize
		def GetInputWordNum(self):
			return self.alignmentSourceWindowSize + self.alignmentTargetWindowSize
		def GetProjectionLayer(self):
			return [self.projectionLayerInputDim, self.projectionLayerOutputDim]
		def GetHiddenLayer1st(self):
			inputNum = (self.alignmentTargetWindowSize + self.alignmentSourceWindowSize) * self.projectionLayerOutputDim
			return [inputNum, self.hiddenLayer1stOutput]
		def GetHiddenLayer2nd(self):
			return[self.hiddenLayer1stOutput, self.hiddenLayer2ndOutput]
		def GetJumpLayer(self):
			return [self.hiddenLayer2ndOutput, self.outputLayerJumpOutput]
		def GetJumpLabelSize(self):
			return self.outputLayerJumpOutput
		def GetJumpLimited(self):
			return self.jumpLimited
		def GetLearningRate(self):
			return self.learningRate

	class AlignmentNeuralNetworkUnitTest:

		def __init__(self):
			
			self.jumpLimited = 3
		def GetJumpLimited(self):
			return self.jumpLimited
		

