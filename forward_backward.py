import math as mt
import numpy as np
import para

class ForwardBackward:

	def __init__(self):
		print('initialize forward backward class')
		self.alignmentNet = para.Para.AlignmentNeuralNetwork()

	def forwardLayer(self, alpha, transition, b):
		# first line:
		# all the entrances to one
		transition = np.array(transition)
		alpha = np.array(alpha)
		b = np.array(b)
		return np.multiply(np.matmul(transition, alpha), b)

	def backwardLayer(self, beta, transition, b):
		transition = np.array(transition).transpose()
		beta = np.array(beta)
		b = np.array(b)
		return np.matmul(transition, np.multiply(beta, b))

	def generateTransitionMatrixInitial(self,prob, num, jumpLim ):
		M = np.resize(prob, (num, num))
		for i in range(num):
			for j in range(num):
				if abs(i - j)>= jumpLim:
					M[i][j] = 0
		return M 


	def calculateForwardBackwardInitialBeta(self, lexicon, targetNum, sourceNum):
		probAlignment = 1.0/sourceNum
		jumpLimited = self.alignmentNet.GetJumpLimited()
		forwardZero = lexicon[0:sourceNum]
		forward = []
		forward.append(forwardZero)
		transitionMatrix = self.generateTransitionMatrixInitial(probAlignment, sourceNum, jumpLimited)
		for i in range(targetNum-1):
			forward[-1] = self.selfArrayNormalization(forward[-1])
			forwardItem = self.forwardLayer(forward[-1], transitionMatrix, lexicon[sourceNum * (i+1): sourceNum*(i+2)])
			forward.append(forwardItem)

		backward = []
		backwardEnd = np.ones( sourceNum )
		backward.append( backwardEnd )

		for i in range(targetNum-1):
			i_ = targetNum - 2 -i
			backward[0] = self.selfArrayNormalization(backward[0])
			backwardItem = self.backwardLayer( backward[0], transitionMatrix, lexicon[sourceNum *(i_+1) : sourceNum*(i_+2)])
			backward.insert(0, backwardItem)

		forward = np.array(forward)
		backward = np.array(backward)
		gamma = np.multiply(forward, backward)
		gamma = self.selfNormalization(gamma)

		alignmentGamma = []

		for t in range(targetNum-1):
			for i in range(sourceNum):
				alignmentGammaItem = []
				for j in range(sourceNum):
					if abs(j-i) >= jumpLimited:
						prob = 0
					else:
						prob =probAlignment
					a_tij = prob
					b_tj = lexicon[(t+1)*sourceNum + j]
					item = forward[t][i] *backward[t+1][j] * a_tij * b_tj
					alignmentGammaItem.append(item)
				alignmentGamma.append(alignmentGammaItem)

		alignmentGamma = self.selfMatrixNormalization(alignmentGamma, sourceNum)

		return gamma, alignmentGamma

	def calculateForwardBackwardInitial(self, lexicon, targetNum, sourceNum):

		probAlignment = 1.0/sourceNum
	
		

		# for limited the jump
		jumpLimited = self.alignmentNet.GetJumpLimited()
		forward = []

		forwardZero =  lexicon[0:sourceNum]


		forward.append( forwardZero )
		for i in range(targetNum-1):
			forwardItem = []

			for j in range(sourceNum):
				item = []
				for j_ in range(sourceNum):
					if abs(j-j_) >= jumpLimited:
						prob = 0
					else:
						prob = probAlignment
					item.append( prob * forward[-1][j_] )
				forwardItem.append(np.sum(item) * lexicon[(i+1)*sourceNum + j])

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
					if abs(j-j_) >= jumpLimited:
						prob = 0
					else:
						prob = probAlignment
					item.append( prob * backward[0][j_] * lexicon[i_*sourceNum +j_])
				backwardItem.append(np.sum(item))
			backward.insert(0, backwardItem)

		forward = np.array(forward)
		backward = np.array(backward)

		gamma = np.multiply(forward, backward)
		gamma = self.selfNormalization(gamma)
		alignmentGamma = []

		for t in range(targetNum-1):
			for i in range(sourceNum):
				alignmentGammaItem = []
				for j in range(sourceNum):
					if abs(j-i) >= jumpLimited:
						prob = 0
					else:
						prob =probAlignment
					a_tij = prob
					b_tj = lexicon[(t+1)*sourceNum + j]
					item = forward[t][i] *backward[t+1][j] * a_tij * b_tj
					alignmentGammaItem.append(item)
				alignmentGamma.append(alignmentGammaItem)

		alignmentGamma = self.selfMatrixNormalization(alignmentGamma, sourceNum)

		return gamma, alignmentGamma

	def calculateForwardBackward(self, lexicon, alignment, targetNum, sourceNum, alignmentInitial = []):

		alignment = np.ndarray.tolist(alignment)
		center = int(mt.floor(float(len(alignment[0]))/2))

		# initial position is 0

		# for limited the jump
		jumpLimited = self.alignmentNet.GetJumpLimited()
		forward = []

		# here we deal with the initial state probabilities
		forwardZero =  lexicon[0:sourceNum]
		
		if len(alignmentInitial) != 0:
			for j in range(min(sourceNum, center)):
				forwardZero[j] *= alignmentInitial[0][center + j]

		# calculate the initial forward value
		forward.append( forwardZero )
		for i in range(targetNum-1):
			forwardItem = []
			for j in range(sourceNum):
				item = []
				for j_ in range(sourceNum):
					if abs(j-j_) >= jumpLimited:
						prob = 0
					else:
						prob = alignment[i*sourceNum + j_][center + j -j_]
					item.append( prob * forward[-1][j_] )
				forwardItem.append(np.sum(item) * lexicon[(i+1)*sourceNum + j])
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
					if abs(j-j_) >= jumpLimited:
						prob = 0
					else:
						prob = alignment[i_*sourceNum + j][center + j_ -j]
					item.append( prob * backward[0][j_] * lexicon[i_*sourceNum +j_])
				backwardItem.append(np.sum(item))
			backward.insert(0, backwardItem)

		forward = np.array(forward)
		backward = np.array(backward)

		gamma = np.multiply(forward, backward)
		gamma = self.selfNormalization(gamma)
		alignmentGamma = []

		for t in range(targetNum-1):
			for i in range(sourceNum):
				alignmentGammaItem = []
				for j in range(sourceNum):
					if abs(j-i) >= jumpLimited:
						prob = 0
					else:
						prob = alignment[t*sourceNum + i][center + j -i]
					a_tij = prob
					b_tj = lexicon[(t+1)*sourceNum + j]
					item = forward[t][i] *backward[t+1][j] * a_tij * b_tj
					alignmentGammaItem.append(item)
				alignmentGamma.append(alignmentGammaItem)

		alignmentGamma = self.selfMatrixNormalization(alignmentGamma, sourceNum)

		return gamma, alignmentGamma

	def selfArrayNormalization(self, anArray):
		sumation = np.sum(anArray)
		if sumation != 0:
			anArray = anArray/sumation
		else:
			anArray = np.ones(len(anArray))/len(anArray)
		return anArray

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

	def installTestEnvironment(self):
		self.alignmentNet = AlignmentNeuralNetworkUnitTest()