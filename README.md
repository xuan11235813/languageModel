# languageModel
machine translation based on tensorflow


* global structure

 - reads data
 -


* runtime environment in RWTH
 -source /work/smt2/dzhu/tensorflow/tensorflow_env/bin/activate 
qsubmit -n zgan_hmm -m 13G -t 160:00:00 -gpu -a cluster-cn-2[26-60] "bash lm_cpu.sh"

* runtime environment at home
 -source ~/tensorflow/bin/activate


u/shin/bin/qsubmit -n test_delete_soonnn_1_ -gpu -a cluster-cn-22* -m 20 -t 16:28:00 "sh tensorflow_test_script.sh"

u/shin/bin/qsubmit -n my_cute_crash_code -m 20 -t 12:48:00 "python main.py"


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

	* Problem:
		how to generate transition probabilities.(That is not problem anymore)


	to 23.07
	* process:
		- every time from main input data( main initialize a process class and input data and implement )
		- generate the sequence at every point
		- input the sequence in network. 
			** from lexicon output ??
			** from alignment output target_step * [state_number * ouput(	101) ]
		- use above data calculate baum welch
		- use previous index(target * state_num) as input,  output as 		indices.
		- new batch

	to 02.09
	* debug
		- bug about baum welch algorithms (without initialial probabilities)
		- calculate the perpelexity
		- run
		- transform to LSTM
		- add BPE

	to 20.20
		- finish LSTM
		- add BPE



questions:

1) lexicon model: previously we use null sequence as our target input when facing the first sample. Here add a 'zero' start for target word sequence (not anymore)

2) alignment model: previously train the initial input distribution as null sequence, now use the 'zero' start (not anymore)
3)how to reuse the tensorflow LSTM network (not anymore)

4) The program will crash when the sentence is too long

5) 17.10 change to BPE and fix the bug