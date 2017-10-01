import math as mt 
import para
import numpy as np

class GenerateSamples:
	def __init__(self):
		print('ready to generate samples')
		self.targetNum = 0
		self.sourceNum = 0
		parameters = para.Para()
		self.lexiconNetPara = parameters.LexiconNeuralNetwork()
		self.alignmentNetPara = parameters.AlignmentNeuralNetwork()
		self.bias = parameters.GetTargetSourceBias()

	# samples with class and inner class index
	def getLexiconSamples( self, sentencePair ):
		self._sentencePair = sentencePair
		self.targetNum = len(sentencePair._target)
		self.sourceNum = len(sentencePair._source)
		samples = []
		labels = []
		for i in range(self.targetNum):
			for j in range(self.sourceNum):
				lexiconSourceStart = int(j - mt.floor(self.lexiconNetPara.GetLexiconSourceWindowSize()/2))
				lexiconSourceEnd = int(lexiconSourceStart + self.lexiconNetPara.GetLexiconSourceWindowSize())
				lexiconTargetStart = int(i - self.lexiconNetPara.GetLexiconTargetWindowSize())
				lexiconTargetEnd = int(lexiconTargetStart + self.lexiconNetPara.GetLexiconTargetWindowSize())
				itemSample = []
				itemLabel = []

				for s in range(lexiconSourceStart, lexiconSourceEnd):
					if (s < 0) | (s >= self.sourceNum):
						itemSample.append(0)
					else:
						itemSample.append(sentencePair._source[s])
				for t in range(lexiconTargetStart, lexiconTargetEnd):
					if (t < 0) | (t >= self.targetNum):
						itemSample.append(0)
					else:
						itemSample.append(sentencePair._target[t] + self.bias)
				itemLabel.append(sentencePair._targetClass[i])
				itemLabel.append(sentencePair._innerClassIndex[i])
				samples.append(itemSample)
				labels.append(itemLabel)
		return samples, labels

	# samples with simplest label (only target index)
	def getSimpleLexiconSamples(self, sentencePair):
		self._sentencePair = sentencePair
		self.targetNum = len(sentencePair._target)
		self.sourceNum = len(sentencePair._source)
		samples = []
		labels = []
		for i in range(self.targetNum):
			for j in range(self.sourceNum):
				lexiconSourceStart = int(j - mt.floor(self.lexiconNetPara.GetLexiconSourceWindowSize()/2))
				lexiconSourceEnd = int(lexiconSourceStart + self.lexiconNetPara.GetLexiconSourceWindowSize())
				lexiconTargetStart = int(i - self.lexiconNetPara.GetLexiconTargetWindowSize())
				lexiconTargetEnd = int(lexiconTargetStart + self.lexiconNetPara.GetLexiconTargetWindowSize())
				itemSample = []

				for s in range(lexiconSourceStart, lexiconSourceEnd):
					if (s < 0) | (s >= self.sourceNum):
						itemSample.append(0)
					else:
						itemSample.append(sentencePair._source[s])
				for t in range(lexiconTargetStart, lexiconTargetEnd):
					if (t < 0) | (t >= self.targetNum):
						itemSample.append(0)
					else:
						itemSample.append(sentencePair._target[t] + self.bias)
				samples.append(itemSample)
				labels.append(i)
		return samples, labels

	def getAlignmentSamples(self, sentencePair):
		self._sentencePair = sentencePair
		self.targetNum = len(sentencePair._target)
		self.sourceNum = len(sentencePair._source)
		samples = []
		initialSample = []
		for i in range(self.targetNum - 1):
			for j in range(self.sourceNum):
				alignmentSourceStart = int(j - mt.floor(self.alignmentNetPara.GetAlignmentSourceWindowSize()/2))
				alignmentSourceEnd = int(alignmentSourceStart + self.alignmentNetPara.GetAlignmentSourceWindowSize())
				alignmentTargetStart = int(i - self.alignmentNetPara.GetAlignmentTargetWindowSize())
				alignmentTargetEnd = int(alignmentTargetStart + self.alignmentNetPara.GetAlignmentTargetWindowSize())
				itemSample = []

				for s in range(alignmentSourceStart, alignmentSourceEnd):
					if (s < 0) | (s >= self.sourceNum):
						itemSample.append(0)
					else:
						itemSample.append(sentencePair._source[s])
				for t in range(alignmentTargetStart, alignmentTargetEnd):
					if (t < 0) | (t >= self.targetNum):
						itemSample.append(0)
					else:
						itemSample.append(sentencePair._target[t] + self.bias)

				samples.append(itemSample)
		sampleSize = self.alignmentNetPara.GetAlignmentSourceWindowSize() + self.alignmentNetPara.GetAlignmentTargetWindowSize()
		for i in range(sampleSize):
			initialSample.append(0)

		return samples, initialSample
		
	def getLabelFromGamma( self, alignmentGamma, lexiconGamma, sentencePair):
		alignmentLabel = np.zeros([(self.targetNum - 1) * self.sourceNum, self.alignmentNetPara.GetJumpLabelSize()])
		initialAlignmentLabel = np.zeros(self.alignmentNetPara.GetJumpLabelSize())
		lexiconLabel = np.zeros([self.targetNum * self.sourceNum, self.lexiconNetPara.GetLabelSize()])
		center = int(self.alignmentNetPara.GetJumpLabelSize()/2)
		
		# create lexicon label
		for i in range(self.targetNum):
			for j in range(self.sourceNum):
				lexiconLabel[i * self.sourceNum + j][sentencePair._targetClass[i]] = lexiconGamma[i][j]

		# create alignment label
		jumpLimited = self.alignmentNetPara.GetJumpLimited()
		for i in range(self.targetNum  - 1):
			for j in range(self.sourceNum):
				for j_ in range(self.sourceNum):
					if abs(j_ -j) <= jumpLimited:
						alignmentLabel[i*self.sourceNum + j][j_ - j + center] = alignmentGamma[i*self.sourceNum +j][j_]

		# initial state probability
		for i in range(center, min(self.alignmentNetPara.GetJumpLabelSize(), center + self.sourceNum)):
			initialAlignmentLabel[i] = lexiconGamma[0][i - center]

		return lexiconLabel, alignmentLabel, initialAlignmentLabel

	def getUnitTestlabel(self, sentencePair):
		alignmentLabel = np.zeros([(self.targetNum - 1) * self.sourceNum, self.alignmentNetPara.GetJumpLabelSize()])
		lexiconLabel = np.zeros([self.targetNum * self.sourceNum, self.lexiconNetPara.GetLabelSize()])
		center = int(self.alignmentNetPara.GetJumpLabelSize()/2)
		for i in range(self.targetNum):
			for j in range(self.sourceNum):
				lexiconLabel[i * self.sourceNum + j][sentencePair._targetClass[i]] = 1
		return lexiconLabel