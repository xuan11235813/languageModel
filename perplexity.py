import numpy as np
import math as mt
import para


class Perplexity:

	def __init__(self):
		self.alignmentNet = para.Para.AlignmentNeuralNetwork()
		self.wordNum = 0.0
		self.logProbabilitySum = 0.0
		self.weight = 1.0

	def reInitialize(self):
		self.wordNum = 0.0
		self.logProbabilitySum = 0.0

	def addLog(self, logValue):
		self.logProbabilitySum += logValue
		if self.logProbabilitySum >= 10e15:
			self.logProbabilitySum *= 0.5
			self.weight *= 0.5

	def calculateLog(self,prob):
		if prob <= 10e-300:
			return mt.log(10e-300)
		elif prob >= 1:
			return 0.0
		else:
			return mt.log(prob)


	def addSequence(self, lexicon, alignment, alignmentInitial, targetNum, sourceNum):
		alignment = np.ndarray.tolist(alignment)
		center = int(mt.floor(float(len(alignment[0]))/2))

		# for limited the jump
		jumpLimited = self.alignmentNet.GetJumpLimited()

		prob = []

		# here we deal with the initial state probabilities
		probZero =  lexicon[0:sourceNum]

		
		if len(alignmentInitial) != 0:
			for j in range(min(sourceNum, center +1)):
				probZero[j] *= alignmentInitial[0][center + j]

		# calculate the initial prob value
		prob.append( probZero )
		for i in range(targetNum-1):
			probItem = []
			for j in range(sourceNum):
				item = []
				for j_ in range(sourceNum):
					if abs(j-j_) >= jumpLimited:
						probabiility = 0
					else:
						probabiility = alignment[i*sourceNum + j_][center + j -j_]
					item.append( probabiility * prob[-1][j_] )
				probItem.append(np.sum(item) * lexicon[(i+1)*sourceNum + j])
			probItem = np.array(probItem)
			#itemMax = np.max(probItem)
			#self.addLog(self.calculateLog(itemMax))
			#probItem = probItem/ itemMax
			prob.append(probItem)

		logProb = self.calculateLog(np.sum(prob[-1]))
		self.addLog(logProb)
		self.wordNum += targetNum

		
			
	def getPerplexity(self):
		if self.wordNum >= 1:
			return - self.logProbabilitySum/self.wordNum
		else:
			return 0



