import math as mt
import lexicon_neural_network as lexiconSet 
import alignment_neural_network as alignmentSet 


class ProcessTraditional:
	def __init__( self, targetClassSetSize ):

		# initialize the networks. lNet for lexicon neural network and 
		# aNet for alignment network.
		 
		lNet = lexiconSet.TraditionalLexiconNet(targetClassSetSize)
		aNet = alignmentSet.TraditionalAlignmentNet()


		print('...')
		print("initialize the process")

	def processBatch( self,  sentencePairBatch):

		print(len(sentencePairBatch))
