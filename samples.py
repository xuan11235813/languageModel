import math as mt 
import para


class GenerateSamples:
	def __init__(self):
		print('ready to generate samples')
		self.targetNum = 0
		self.sourceNum = 0
		self.lexiconNetPara = para.Para.LexiconNeuralNetwork()
		self.alignmentNetPara = para.Para.AlignmentNeuralNetwork()

	
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

	def getAlignmentSamples(self, sentencePair):
		self._sentencePair = sentencePair
		self.targetNum = len(sentencePair._target)
		self.sourceNum = len(sentencePair._source)
		samples = []
		labels = []
		for i in range(self.targetNum):
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