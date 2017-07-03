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
	toFollowingBatch()

	3. add table lookup tensorflow variables to lexicon_neural_network.py so that it can produce the concatenate vector

	result: can run the network