import math as mt
import numpy as np

class ForwardBackward:

	def __init__(self):
		print('initialize forward backward class')


	def calculateForwardBackward(self, lexicon, alignment, targetNum, sourceNum):

		alignment = np.ndarray.tolist(alignment)
		center = int(mt.floor(float(len(alignment[0]))/2))
		forward = []
		forwardZero =  lexicon[0:sourceNum]
		forward.append( forwardZero )
		for i in range(targetNum-1):
			forwardItem = []
			for j in range(sourceNum):
				item = []
				for j_ in range(sourceNum):
					item.append( alignment[i*sourceNum + j_][center + j -j_]*forward[-1][j_] )
				forwardItem.append(np.sum(item))
			forward.append(forwardItem)

		backward = []
		backwardEnd = np.ndarray.tolist(np.ones( sourceNum ))
		backward.append( backwardEnd )
		for i in range( targetNum-1):
			backwardItem = []
			for j in range( sourceNum ):
				item = []
				for j_ in range( sourceNum ):
					i_ = targetNum - 2  -i 
					item.append( alignment[i_*sourceNum + j][center + j_ -j] * backward[0][j_] )
				backwardItem.append(np.sum(item))
			backward.insert(0, backwardItem)

		forward = np.array(forward)
		backward = np.array(backward)
		gamma = forward * backward
		gamma = self.selfNormilization(gamma)
		
		alignmentProb = []

		for i in range(targetNum-1):
			



		return lexicon, alignment

	def selfNormilization(self, anArray):
		for i in range(len(anArray)):
			listItemSum = np.sum(anArray[i])
			if listItemSum != 0:
				anArray[i] = anArray[i]/listItemSum
			else:
				anArray[i] = np.ones(len(anArray[i]))/len(anArray[i])
		return anArray