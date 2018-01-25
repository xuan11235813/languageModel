import numpy as np 
import para
import math as mt
import os.path
import lexicon_neural_network as lexiconSet
import alignment_neural_network as alignmentSet 

class LanguageNetwork:
	def __init__(self):

		continue_pre = para.Para().SetTranslationMode()
		
		# initialize the networks. lNet for lexicon neural network and 
		# aNet for alignment network.
		 
		self.lNet = lexiconSet.LSTMLexiconNet(continue_pre)
		self.aNet = alignmentSet.LSTMAlignmentNet(continue_pre)


    def getAlignmentJumpDistribution(self, _sourceTarget, _sourceNum, _targetNum):
    	output = self.aNet.networkTranslationPrognose(_sourceTarget, _sourceNum, _targetNum):
    	return output
    
    def getLexiconDistribution(self, sourceTarget, _sourceNum, _targetNum):
    	output = self.lNet.networkTranslationPrognose(_sourceTarget, _sourceNum, _targetNum):
    	return output

    def getAlignmentInitialDistribution(self):
    	output = self.aNet.networkTranslationInitial()
    	return output

    