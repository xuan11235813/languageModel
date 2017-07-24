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
		gamma = self.selfNormalization(gamma)
		
		alignmentGamma = []

		for t in range(targetNum-1):
			for i in range(sourceNum):
				alignmentGammaItem = []
				for j in range(sourceNum):
					a_tij = alignment[t*sourceNum + i][center + j -i]
					b_tj = lexicon[(t+1)*sourceNum + j]
					item = forward[t][i] *backward[t+1][j] * a_tij * b_tj
					alignmentGammaItem.append(item)
				alignmentGamma.append(alignmentGammaItem)

		alignmentGamma = self.selfMatrixNormalization(alignmentGamma, sourceNum)
		return gamma, alignmentGamma

	def selfNormalization( self, anArray ):
		for i in range(len(anArray)):
			listItemSum = np.sum(anArray[i])
			if listItemSum != 0:
				anArray[i] = anArray[i]/listItemSum
			else:
				anArray[i] = np.ones(len(anArray[i]))/len(anArray[i])
		return anArray

	def selfMatrixNormalization( self, anArray, step ):
		anArray = np.array(anArray)
		frame = len(anArray)/step
		for i in range(frame):
			matrixSum = np.sum(anArray[(i*step):((i+1)*step)])
			if matrixSum != 0:
				anArray[(i*step):((i+1)*step)] = anArray[(i*step):((i+1)*step)]/matrixSum
		return anArray