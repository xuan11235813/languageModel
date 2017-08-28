import numpy as np
import math as mt
import para


class Perplexity:

	def __init__(self):
		self.alignmentNet = para.Para.AlignmentNeuralNetwork()
		self.wordNum = 0.0
		self.logProbabilitySum = 0.0

	def reInitialize(self):
		self.wordNum = 0.0
		self.logProbabilitySum = 0.0

	def addSequence(self, lexicon, alignment, targetNum, sourceNum):
		alignment = np.ndarray.tolist(alignment)
		center = int(mt.floor(float(len(alignment[0]))/2))

		# for limited the jump
		jumpLimited = self.alignmentNet.GetJumpLimited()
		

			
	def getPerplexity(self):
		if self.wordNum >= 1:
			return - self.logProbabilitySum/self.wordNum
		else:
			return 0



