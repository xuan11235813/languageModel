import math as mt
import lexicon_neural_network as lexiconSet 
import alignment_neural_network as alignmentSet 
import samples

class ProcessTraditional:
	def __init__( self, targetClassSetSize ):

		# initialize the networks. lNet for lexicon neural network and 
		# aNet for alignment network.
		 
		self.lNet = lexiconSet.TraditionalLexiconNet(targetClassSetSize)
		self.aNet = alignmentSet.TraditionalAlignmentNet()
		self.generator = samples.GenerateSamples()


		print('...')
		print("initialize the process")

	def processBatch( self,  sentencePairBatch):

		for i in range(len(sentencePairBatch)):
			sentencePair = sentencePairBatch[i]
			samplesLexicon, labelsLexicon = self.generator.getLexiconSamples( sentencePair )
			samplesAlignment, labelsAlignment = self.generator.getAlignmentSamples( sentencePair )
			outputLexicon = self.lNet.networkPrognose(samplesLexicon, labelsLexicon)
			outputAlignment = self.aNet.networkPrognose(samplesAlignment)
			# calculate forward backward
			# training
			print(i)
