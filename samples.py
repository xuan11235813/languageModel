import math as mt 
import para
import numpy as np

class GenerateSamples:
	def __init__(self):
		print('ready to generate samples')
		self.targetNum = 0
		self.sourceNum = 0
		self.lexiconNetPara = para.Para.LexiconNeuralNetwork()
		self.alignmentNetPara = para.Para.AlignmentNeuralNetwork()

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
						itemSample.append(sentencePair._target[t])
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
						itemSample.append(sentencePair._target[t])
				samples.append(itemSample)
				labels.append(i)
		return samples, labels


	def getAlignmentSamples(self, sentencePair):
		self._sentencePair = sentencePair
		self.targetNum = len(sentencePair._target)
		self.sourceNum = len(sentencePair._source)
		samples = []
		labels = []
		for i in range(self.targetNum - 1):
			for j in range(self.sourceNum):
				alignmentSourceStart = int(j - mt.floor(self.alignmentNetPara.GetAlignmentSourceWindowSize()/2))
				alignmentSourceEnd = int(alignmentSourceStart + self.alignmentNetPara.GetAlignmentSourceWindowSize())
				alignmentTargetStart = int(i - self.alignmentNetPara.GetAlignmentTargetWindowSize())
				alignmentTargetEnd = int(alignmentTargetStart + self.alignmentNetPara.GetAlignmentTargetWindowSize())
				itemSample = []
				itemLabel = []

				for s in range(alignmentSourceStart, alignmentSourceEnd):
					if (s < 0) | (s >= self.sourceNum):
						itemSample.append(0)
					else:
						itemSample.append(sentencePair._source[s])
				for t in range(alignmentTargetStart, alignmentTargetEnd):
					if (t < 0) | (t >= self.targetNum):
						itemSample.append(0)
					else:
						itemSample.append(sentencePair._target[t])
				itemLabel.append(sentencePair._targetClass[i])
				itemLabel.append(sentencePair._innerClassIndex[i])
				samples.append(itemSample)
				labels.append(itemLabel)
		return samples, labels
		
	def getLabelFromGamma( self, alignmentGamma, lexiconGamma, sentencePair):
		alignmentLabel = np.zeros([(self.targetNum - 1) * self.sourceNum, self.alignmentNetPara.GetJumpLabelSize()])
		lexiconLabel = np.zeros([self.targetNum * self.sourceNum, self.lexiconNetPara.GetClassLabelSize()])
		center = int(self.alignmentNetPara.GetJumpLabelSize()/2)
		for i in range(self.targetNum):
			for j in range(self.sourceNum):
				lexiconLabel[i * self.sourceNum + j][sentencePair._targetClass[i]] = lexiconGamma[i][j]

		jumpLimited = self.alignmentNetPara.GetJumpLimited()
		for i in range(self.targetNum  - 1):
			for j in range(self.sourceNum):
				for j_ in range(self.sourceNum):
					if abs(j_ -j) <= jumpLimited:
						alignmentLabel[i*self.sourceNum + j][j_ - j + center] = alignmentGamma[i*self.sourceNum +j][j_]


		return lexiconLabel, alignmentLabel

	def getUnitTestlabel(self, sentencePair):
		alignmentLabel = np.zeros([(self.targetNum - 1) * self.sourceNum, self.alignmentNetPara.GetJumpLabelSize()])
		lexiconLabel = np.zeros([self.targetNum * self.sourceNum, self.lexiconNetPara.GetClassLabelSize()])
		center = int(self.alignmentNetPara.GetJumpLabelSize()/2)
		for i in range(self.targetNum):
			for j in range(self.sourceNum):
				lexiconLabel[i * self.sourceNum + j][sentencePair._targetClass[i]] = 1
		return lexiconLabel