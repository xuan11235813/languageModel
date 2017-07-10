# languageModel
machine translation based on tensorflow


* global structure

 - reads data
 -


* runtime environment in RWTH
 -source /work/smt2/dzhu/tensorflow/tensorflow_env/bin/activate 


* runtime environment at home
 -source ~/tensorflow/bin/activate


for the following week to 09.07
	1. add the index series function findSample() to Data.py
	input:  source index
			target index
	output: series with window
	

	2. add modify the Data.py so that can regenerate 128 sentences and continue read the file
	refreshNewBatch()
	# DONE at 05.07

	3. add table lookup tensorflow variables to lexicon_neural_network.py so that it can produce the concatenate vector

	result: can run the network

	Will be done in 9 days.

	* lexicon_model:
		- Class
		- input Data, generate one table/array of f(ei|...) as emission probability stored in Data, function only output one table probability
	* alignmenta_model:
		- Class
		- input Data, generate transition probabilities
	* Process:
		- Save gamma, 
		- use gamma to update neural network, 
		- generate the training data, 
		- input data into nueral network model ,
		- have two native neural network

	Problem:
		how to generate transition probabilities.