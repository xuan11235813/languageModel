import math as mt
import lexicon_neural_network as lexiconSet 
import alignment_neural_network as alignmentSet 
import samples
import forward_backward as fb 
import para
class ProcessTraditional:
	def __init__( self ):

		# configure restart or continue
		continue_pre = para.Para().ContinueOrRestart()
		
		# initialize the networks. lNet for lexicon neural network and 
		# aNet for alignment network.
		 
		self.lNet = lexiconSet.TraditionalLexiconNet(continue_pre)
		self.aNet = alignmentSet.TraditionalAlignmentNet(continue_pre)
		self.generator = samples.GenerateSamples()
		self.forwardBackward = fb.ForwardBackward()

		self.globalSentenceNum = 0;

		print('...')
		print("initialize the process finish")

	def recordNetwork(self):
		self.lNet.saveMatrixToFile()
		self.aNet.saveMatrixToFile()
		
	def processBatchWithBaumWelch( self, sentencePairBatch):
		averageCostAlignment = 0;
		averageCostLexicon = 0;

		for i in range(len(sentencePairBatch)):
			sentencePair = sentencePairBatch[i]
			targetNum, sourceNum = sentencePair.getSentenceSize()
			samplesLexicon, labelsLexicon = self.generator.getSimpleLexiconSamples( sentencePair )
			samplesAlignment, labelsAlignment = self.generator.getAlignmentSamples( sentencePair )
			outputLexicon = sentencePair.getIBMLexiconInitialData()
			gamma, alignmentGamma = self.forwardBackward.calculateForwardBackwardInitial( outputLexicon, targetNum, sourceNum )
			lexiconLabel, alignmentLabel = self.generator.getLabelFromGamma(alignmentGamma, gamma, sentencePair)
			costLexicon = self.lNet.trainingBatch(samplesLexicon, lexiconLabel)
			costAlignment = self.aNet.trainingBatch(samplesAlignment, alignmentLabel)
			averageCostLexicon = (averageCostLexicon * i + costLexicon)/(i+1)
			averageCostAlignment = (averageCostAlignment * i + costAlignment)/(i+1)
			self.globalSentenceNum += 1

		print('costLexicon:   '+ repr(averageCostLexicon))
		print('costAlignment: '+ repr(averageCostAlignment))


	def processBatch( self,  sentencePairBatch):

		averageCostAlignment = 0;
		averageCostLexicon = 0;

		for i in range(len(sentencePairBatch)):
			sentencePair = sentencePairBatch[i]
			targetNum, sourceNum = sentencePair.getSentenceSize()
			samplesLexicon, labelsLexicon = self.generator.getSimpleLexiconSamples( sentencePair )
			samplesAlignment, labelsAlignment = self.generator.getAlignmentSamples( sentencePair )
			outputLexicon = self.lNet.networkPrognose(samplesLexicon, labelsLexicon)
			outputAlignment = self.aNet.networkPrognose(samplesAlignment)
			gamma, alignmentGamma = self.forwardBackward.calculateForwardBackward( outputLexicon, outputAlignment, targetNum, sourceNum )
			lexiconLabel, alignmentLabel = self.generator.getLabelFromGamma(alignmentGamma, gamma, sentencePair)
			costLexicon = self.lNet.trainingBatch(samplesLexicon, lexiconLabel)
			costAlignment = self.aNet.trainingBatch(samplesAlignment, alignmentLabel)			
			averageCostLexicon = (averageCostLexicon * i + costLexicon)/(i+1)
			averageCostAlignment = (averageCostAlignment * i + costAlignment)/(i+1)
			self.globalSentenceNum += 1

		print('costLexicon:   '+ repr(averageCostLexicon))
		print('costAlignment: '+ repr(averageCostAlignment))

	def processUnitLexiconTest(self, sentencePairBatch):
		averageCostLexicon = 0;

		sentencePair = sentencePairBatch[0]
		targetNum, sourceNum = sentencePair.getSentenceSize()
		samplesLexicon, labelsLexicon = self.generator.getSimpleLexiconSamples( sentencePair )
		lexiconLabel= self.generator.getUnitTestlabel(sentencePair)
		for i in range(30):
			costLexicon = self.lNet.trainingBatch(samplesLexicon, lexiconLabel)
			print(costLexicon)

	def processPerplexity(self, sentencePairBatch):

		# initialize a perplexity
		for i in range(len(sentencePairBatch)):
			sentencePair = sentencePairBatch[i]
			# calculate the network result
			# input the perplexity

		# get result