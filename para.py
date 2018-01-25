
class Para:
	def __init__(self):

		# general configuration
		# if read network from files
		self.continue_pre = 1
		#self.networkStoragePath = '/u/zgan/Desktop/HMMLM/languageModel/data/network/'
		self.networkStoragePath = 'data/network/'
		# the following variables are used for previous data
		# input class and will be deprecated later
		# for source data file
		#self.sourceVocabFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/sourceVocab"
		#self.targetVocabFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/engClass"
		#self.trainingDataFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/dev.train"
		#self.IBMDataFilePath = '/u/zgan/Desktop/HMMLM/languageModel/data/prob'		
		#self.measureDataFilePath = '/u/zgan/Desktop/HMMLM/languageModel/data/dev'
		self.sourceVocabFilePath = "data/sourceVocab"
		self.targetVocabFilePath = "data/engClass"
		self.trainingDataFilePath = "data/dev.train"
		self.IBMDataFilePath = 'data/prob'		
		self.measureDataFilePath = 'data/dev'


		# these paths are for bpe
		#self.trainingSourceDataFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/train_german.bpe"
		#self.trainingTargetDataFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/train_english.bpe"
		#self.IBM1DataFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/probIBM1"
		#self.IBM2DataFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/probIBM2"
		#self.measureSourceDataFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/dev_german.bpe"
		#self.measureTargetDataFilePath = "/u/zgan/Desktop/HMMLM/languageModel/data/dev_english.bpe"

		self.trainingSourceDataFilePath = "data/train_german.bpe"
		self.trainingTargetDataFilePath = "data/train_english.bpe"
		self.IBM1DataFilePath = "data/probIBM1"
		self.IBM2DataFilePath = "data/probIBM2"
		self.measureSourceDataFilePath = "data/dev_german.bpe"
		self.measureTargetDataFilePath = "data/dev_english.bpe"

		#training property
		self.batchSize = 40
		self.testBatchSize = 40

	# path for dataInput.py
	def GetBatchSize(self):
		return self.batchSize
	def GetSourceTrainingFilePath(self):
		return self.trainingSourceDataFilePath
	def GetTargetTrainingFilePath(self):
		return self.trainingTargetDataFilePath
	def GetIBMDataFile1(self):
		return self.IBM1DataFilePath
	def GetIBMDataFile2(self):
		return self.IBM2DataFilePath
	def GetSourceMeasureFilePath(self):
		return self.measureSourceDataFilePath
	def GetTargetMeasureFilePath(self):
		return self.measureTargetDataFilePath

	# path for data.py, will be deprecated later
	def GetSourceVocabFilePath(self):
		return self.sourceVocabFilePath
	def GetTargetVocabFilePath(self):
		return self.targetVocabFilePath
	def GetTrainingDataFilePath(self):
		return self.trainingDataFilePath
	def GetIBMFilePath(self):
		return self.IBMDataFilePath
	def GetMeasureDataFilePath(self):
		return self.measureDataFilePath

	# global variables and path
	def SetTranslationMode(self):
		return 1
	def ContinueOrRestart(self):
		return self.continue_pre
	def GetNetworkStoragePath(self):
		return self.networkStoragePath	
	def GetTestBatchSize(self):
		return self.testBatchSize
	def ReadTrainingFile(self):
		return 0
	def ReadTestFile(self):
		return 1
	def GetTargetSourceBias(self):
		return 41000
	def GetLSTMBatchSize(self):
		return 1


	#for lexicon neural network
	class LexiconNeuralNetwork:

		def __init__(self, mode = ''):

			# In order to make the code readable, no nameError will
			# be held here.
			# if somebody use the wrong mode to find the parameters
			# the program will crash immediately with a nameError "no attribute"

			if mode == 'lstm':

				self.projectionLayerInputDim = 80000
				self.projectionLayerOutputDim = 200
				# 200 for source, 200 for target
				self.hiddenLayer1stInput = 400
				self.hiddenLayer1stOutput = 1000
				self.hiddenLayer2ndOutput = 500
				self.outputLayerOutput = 12000


			else:
				# will be deprecated later
				self.lexiconSourceWindowSize = 5
				self.lexiconTargetWindowSize = 3
				self.projectionLayerInputDim = 70681
				self.projectionLayerOutputDim = 200
				self.hiddenLayer1stOutput = 1000
				self.hiddenLayer2ndOutput = 500
				self.outputLayerOutput = 12000
				self.outputLayerWordOutput = 40681

			self.learningRate = 0.001
			self.mode = mode


		def GetLexiconSourceWindowSize(self):			       	
			return self.lexiconSourceWindowSize
		def GetLexiconTargetWindowSize(self):
			return self.lexiconTargetWindowSize
		def GetInputWordNum(self):
			return self.lexiconSourceWindowSize + self.lexiconTargetWindowSize
		def GetProjectionLayer(self):
			return [self.projectionLayerInputDim, self.projectionLayerOutputDim]
		def GetHiddenLayer1st(self):
			if self.mode == 'lstm':
				return [self.hiddenLayer1stInput, self.hiddenLayer1stOutput]
			else:
				inputNum = (self.lexiconTargetWindowSize + self.lexiconSourceWindowSize) * self.projectionLayerOutputDim
				return [inputNum, self.hiddenLayer1stOutput]
		def GetHiddenLayer2nd(self):
			return[self.hiddenLayer1stOutput, self.hiddenLayer2ndOutput]
		def GetOutputLayer(self):
			return [self.hiddenLayer2ndOutput, self.outputLayerOutput]
		def GetLabelSize(self):
			return self.outputLayerOutput
		def GetLearningRate(self):
			return self.learningRate

	#for lexicon neural network
	class AlignmentNeuralNetwork:

		def __init__(self, mode = ''):

			# In order to make the code readable, no nameError will
			# be held here.
			# if somebody use the wrong mode to find the parameters
			# the program will crash immediately with a nameError "no attribute"

			if mode == 'lstm':

				self.projectionLayerInputDim = 80000
				self.projectionLayerOutputDim = 200
				# 200 for source, 200 for target
				self.hiddenLayer1stInput = 400
				self.hiddenLayer1stOutput = 1000
				self.hiddenLayer2ndOutput = 500
				self.outputLayerOutput = 2000
				self.outputLayerJumpOutput = 101
				self.jumpLimited = 50


			else:
				# will be deprecated later
				self.alignmentSourceWindowSize = 5
				self.alignmentTargetWindowSize = 3
				self.projectionLayerInputDim = 70681
				self.projectionLayerOutputDim = 200
				self.hiddenLayer1stOutput = 1000
				self.hiddenLayer2ndOutput = 500
				self.outputLayerJumpOutput = 101
				self.jumpLimited = 50

			self.learningRate = 0.001
			self.mode = mode
			

		def GetAlignmentSourceWindowSize(self):			       	
			return self.alignmentSourceWindowSize
		def GetAlignmentTargetWindowSize(self):
			return self.alignmentTargetWindowSize
		def GetInputWordNum(self):
			return self.alignmentSourceWindowSize + self.alignmentTargetWindowSize
		def GetProjectionLayer(self):
			return [self.projectionLayerInputDim, self.projectionLayerOutputDim]
		def GetHiddenLayer1st(self):
			if self.mode == 'lstm':
				return [self.hiddenLayer1stInput, self.hiddenLayer1stOutput]
			else:
				inputNum = (self.alignmentSourceWindowSize + self.alignmentTargetWindowSize) * self.projectionLayerOutputDim
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
		

