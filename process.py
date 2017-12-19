import math as mt
import lexicon_neural_network as lexiconSet
import alignment_neural_network as alignmentSet 
import samples
import forward_backward as fb 
import para
import perplexity as pp
import printLog



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
		self.perplexity = pp.Perplexity()
		self.log = printLog.Log()

		self.globalSentenceNum = 0;

		print('...')
		print("initialize the process finish")

	def recordNetwork(self):
		self.lNet.saveMatrixToFile()
		self.aNet.saveMatrixToFile()
		
		
	def processBatchWithBaumWelch( self, sentencePairBatch):
		averageCostAlignment = 0;
		averageCostLexicon = 0;
		averageCostAlignmentInitialState = 0;

		for i in range(len(sentencePairBatch)):
			# read a sentence
			sentencePair = sentencePairBatch[i]

			# get the size of target and source
			targetNum, sourceNum = sentencePair.getSentenceSize()

			# generate lexicon training samples
			samplesLexicon, labelsLexicon = self.generator.getSimpleLexiconSamples( sentencePair )

			# generate alignment samples, sampleInitial is the initial distribution of path (initial state distribution)
			samplesAlignment, sampleInitial = self.generator.getAlignmentSamples( sentencePair )
			
			# use IBM to get initial lexicon data
			outputLexicon = sentencePair.getIBMLexiconInitialData()

			# use baum-welch algorithms to optimize the lexicon probability and alignment probability
			gamma, alignmentGamma = self.forwardBackward.calculateForwardBackwardInitial( outputLexicon, targetNum, sourceNum )
			
			# generate lables from gamma which obtained from baum-welch algorithms
			lexiconLabel, alignmentLabel, alignmentLabelInitial = self.generator.getLabelFromGamma(alignmentGamma, gamma, sentencePair)
			
			# use data training lexicon neural network
			costLexicon = self.lNet.trainingBatch(samplesLexicon, lexiconLabel)
			
			# use data training alignment neural network
			# costAlignment = self.aNet.trainingBatch(samplesAlignment, alignmentLabel)
			# costInitial = self.aNet.trainingInitialState(sampleInitial, alignmentLabelInitial)
			costAlignment = self.aNet.trainingBatchWithInitial(samplesAlignment, alignmentLabel, sampleInitial, alignmentLabelInitial)
			
			# output the result
			averageCostLexicon = (averageCostLexicon * i + costLexicon)/(i+1)
			averageCostAlignment = (averageCostAlignment * i + costAlignment)/(i+1)
			# averageCostAlignmentInitialState = (averageCostAlignmentInitialState * i + costInitial)/(i+1)
			self.globalSentenceNum += 1

		self.log.writeSequence('costLexicon:  '+ repr(averageCostLexicon))
		self.log.writeSequence('costAlignment: '+ repr(averageCostAlignment))
		# self.log.writeSequence('costInitial: '+ repr(averageCostAlignmentInitialState))


	def processBatch( self,  sentencePairBatch):

		averageCostAlignment = 0;
		averageCostLexicon = 0;

		for i in range(len(sentencePairBatch)):
			# read a sentence
			sentencePair = sentencePairBatch[i]

			# get the size of target and source
			targetNum, sourceNum = sentencePair.getSentenceSize()

			# generate lexicon training samples
			samplesLexicon, labelsLexicon = self.generator.getSimpleLexiconSamples( sentencePair )
			
			# generate alignment samples, sampleInitial is the initial distribution of path (initial state distribution)
			samplesAlignment, sampleInitial = self.generator.getAlignmentSamples( sentencePair )
			
			# use network to produce the probabilities
			outputLexicon = self.lNet.networkPrognose(samplesLexicon, labelsLexicon)
			outputAlignment, outputAlignmentInitial = self.aNet.networkPrognose(samplesAlignment, sampleInitial)

			# use baum welch to optimize the probabilities
			gamma, alignmentGamma = self.forwardBackward.calculateForwardBackward( outputLexicon, outputAlignment, targetNum, sourceNum, outputAlignmentInitial )
			
			# gamma[0] is our updated initial state probabilities
			lexiconLabel, alignmentLabel, alignmentLabelInitial  = self.generator.getLabelFromGamma(alignmentGamma, gamma, sentencePair)
			
			# use data training lexicon neural network
			costLexicon = self.lNet.trainingBatch([samplesLexicon], lexiconLabel, sourceNum, targetNum)
			
			# use data training alignment neural network
			costAlignment = self.aNet.trainingBatchWithInitial([samplesAlignment], alignmentLabel, alignmentLabelInitial)
			
			# output the result
			averageCostLexicon = (averageCostLexicon * i + costLexicon)/(i+1)
			averageCostAlignment = (averageCostAlignment * i + costAlignment)/(i+1)
			# averageCostAlignmentInitialState = (averageCostAlignmentInitialState * i + costInitial)/(i+1)
			self.globalSentenceNum += 1

		self.log.writeSequence('costLexicon:  '+ repr(averageCostLexicon))
		self.log.writeSequence('costAlignment: '+ repr(averageCostAlignment))
		# self.log.writeSequence('costInitial: '+ repr(averageCostAlignmentInitialState))
	
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
		self.perplexity.reInitialize()

		for i in range(len(sentencePairBatch)):
			sentencePair = sentencePairBatch[i]
			targetNum, sourceNum = sentencePair.getSentenceSize()
			samplesLexicon, labelsLexicon = self.generator.getSimpleLexiconSamples( sentencePair )
			samplesAlignment, sampleInitial = self.generator.getAlignmentSamples( sentencePair )
			outputLexicon = self.lNet.networkPrognose(samplesLexicon, labelsLexicon)
			outputAlignment, outputAlignmentInitial = self.aNet.networkPrognose(samplesAlignment, sampleInitial)
			self.perplexity.addSequence(outputLexicon, outputAlignment, outputAlignmentInitial, targetNum, sourceNum)
		
		# get result
		print('perplexity:  '+ repr(self.perplexity.getPerplexity()))
		self.log.writeSequence('perplexity:  '+ repr(self.perplexity.getPerplexity()))

class ProcessLSTM:
	def __init__( self ):

		# configure restart or continue
		continue_pre = para.Para().ContinueOrRestart()
		
		# initialize the networks. lNet for lexicon neural network and 
		# aNet for alignment network.
		 
		self.lNet = lexiconSet.LSTMLexiconNet(continue_pre)
		self.aNet = alignmentSet.LSTMAlignmentNet(continue_pre)
		self.generator = samples.GenerateSamples('lstm')
		self.forwardBackward = fb.ForwardBackward()
		self.perplexity = pp.Perplexity()
		self.log = printLog.Log()

		self.globalSentenceNum = 0;

		print('...')
		print("initialize the process finish")

	def recordNetwork(self):
		self.lNet.saveMatrixToFile()
		self.aNet.saveMatrixToFile()
		
		
	
	def processBatchWithBaumWelch(self, sentencePairBatch):
		averageCostAlignment = 0;
		averageCostLexicon = 0;
		averageCostAlignmentInitialState = 0;
		for i in range(len(sentencePairBatch)):
		#for i in range(10):
			#print(i)
			# read a sentence
			sentencePair = sentencePairBatch[i]

			# get the size of target and source
			targetNum, sourceNum = sentencePair.getSentenceSize()

			# generate lexicon training samples
			samplesLexicon, labelsLexicon = self.generator.getLSTMLexiconSample( sentencePair )

			# generate alignment samples, sampleInitial is the initial distribution of path (initial state distribution)
			samplesAlignment = self.generator.getLSTMAlignmentSample( sentencePair )
			
			# use IBM to get initial lexicon data
			outputLexicon = sentencePair.getIBMLexiconInitialData()

			# use baum-welch algorithms to optimize the lexicon probability and alignment probability
			gamma, alignmentGamma = self.forwardBackward.calculateForwardBackwardInitial( outputLexicon, targetNum, sourceNum )
			
			# generate lables from gamma which obtained from baum-welch algorithms
			lexiconLabel, alignmentLabel, alignmentLabelInitial = self.generator.getLabelFromGamma(alignmentGamma, gamma, sentencePair)
			
			# use data training lexicon neural network
			costLexicon = self.lNet.trainingBatch([samplesLexicon], lexiconLabel, sourceNum, targetNum)

			# use data training alignment neural network
			costAlignment = self.aNet.trainingBatchWithInitial([samplesAlignment], alignmentLabel, alignmentLabelInitial, sourceNum, targetNum)
			#costAlignment = 0
			# output the result
			averageCostLexicon = (averageCostLexicon * i + costLexicon)/(i+1)
			averageCostAlignment = (averageCostAlignment * i + costAlignment)/(i+1)
			# averageCostAlignmentInitialState = (averageCostAlignmentInitialState * i + costInitial)/(i+1)
			self.globalSentenceNum += 1


		self.log.writeSequence('costLexicon:  '+ repr(averageCostLexicon))
		self.log.writeSequence('costAlignment: '+ repr(averageCostAlignment))


	def processBatch( self,  sentencePairBatch):

		averageCostAlignment = 0;
		averageCostLexicon = 0;

		for i in range(len(sentencePairBatch)):
			# read a sentence
			sentencePair = sentencePairBatch[i]

			# get the size of target and source
			targetNum, sourceNum = sentencePair.getSentenceSize()

			# generate lexicon training samples
			samplesLexicon, labelsLexicon = self.generator.getLSTMLexiconSample( sentencePair )

			# generate alignment samples, sampleInitial is the initial distribution of path (initial state distribution)
			samplesAlignment = self.generator.getLSTMAlignmentSample( sentencePair )
			
			# use network to produce the probabilities
			outputLexicon = self.lNet.networkPrognose([samplesLexicon], labelsLexicon, sourceNum, targetNum)
			outputAlignment, outputAlignmentInitial = self.aNet.networkPrognose([samplesAlignment], sourceNum, targetNum)

			# use baum welch to optimize the probabilities
			gamma, alignmentGamma = self.forwardBackward.calculateForwardBackward( outputLexicon, outputAlignment, targetNum, sourceNum, outputAlignmentInitial )
			
			# gamma[0] is our updated initial state probabilities
			lexiconLabel, alignmentLabel, alignmentLabelInitial  = self.generator.getLabelFromGamma(alignmentGamma, gamma, sentencePair)
			

			# train the network
			costLexicon = self.lNet.trainingBatch([samplesLexicon], lexiconLabel, sourceNum, targetNum)
			
			# use data training alignment neural network
			costAlignment = self.aNet.trainingBatchWithInitial([samplesAlignment], alignmentLabel, alignmentLabelInitial, sourceNum, targetNum)
			
			# output the result
			averageCostLexicon = (averageCostLexicon * i + costLexicon)/(i+1)
			averageCostAlignment = (averageCostAlignment * i + costAlignment)/(i+1)
			# averageCostAlignmentInitialState = (averageCostAlignmentInitialState * i + costInitial)/(i+1)
			self.globalSentenceNum += 1

		self.log.writeSequence('costLexicon:  '+ repr(averageCostLexicon))
		self.log.writeSequence('costAlignment: '+ repr(averageCostAlignment))
		# self.log.writeSequence('costInitial: '+ repr(averageCostAlignmentInitialState))
	def processUnitLexiconTest(self, sentencePairBatch):
		averageCostLexicon = 0;

		sentencePair = sentencePairBatch[0]
		targetNum, sourceNum = sentencePair.getSentenceSize()
		samplesLexicon, labelsLexicon = self.generator.getLSTMLexiconSample( sentencePair )
		lexiconLabel= self.generator.getUnitTestlabel(sentencePair)
		for i in range(30):
			costLexicon = self.lNet.trainingBatch([samplesLexicon], lexiconLabel, sourceNum, targetNum)
			print(costLexicon)

	def processPerplexity(self, sentencePairBatch):

		# initialize a perplexity
		self.perplexity.reInitialize()

		for i in range(len(sentencePairBatch)):
			sentencePair = sentencePairBatch[i]
			targetNum, sourceNum = sentencePair.getSentenceSize()
			samplesLexicon, labelsLexicon = self.generator.getLSTMLexiconSample( sentencePair )
			samplesAlignment = self.generator.getLSTMAlignmentSample( sentencePair )
			outputLexicon = self.lNet.networkPrognose([samplesLexicon], labelsLexicon, sourceNum, targetNum)
			outputAlignment, outputAlignmentInitial = self.aNet.networkPrognose([samplesAlignment], sourceNum, targetNum)

			self.perplexity.addSequence(outputLexicon, outputAlignment, outputAlignmentInitial, targetNum, sourceNum)
		
		# get result
		print('perplexity:  '+ repr(self.perplexity.getPerplexity()))
		self.log.writeSequence('perplexity:  '+ repr(self.perplexity.getPerplexity()))