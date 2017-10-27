import sys
import para
import os

probPath = "data/ibm1"
target1Path = 'data/probIBM1'
target2Path = 'data/probIBM2'
sourcePath = os.path.join(os.path.dirname(__file__), probPath)

fileSource = open(sourcePath, 'r')

sourceProb = []
target1Prob = []
target2Prob = []
for line in fileSource:
	sourceProb.append(line.rstrip())

print('read ' + repr(len(sourceProb)) + ' lines')

len1 = int(len(sourceProb)/2)
len2 = len(sourceProb) - len1;

print('probIBM1 will have '+ repr(len1) + ' lines')
print('probIBM2 will have '+ repr(len2) + ' lines')

for i in range(len1):
	target1Prob.append(sourceProb[i])

for i in range(len2):
	target2Prob.append(sourceProb[i + len1])

print(sourceProb[2572437])
print(target1Prob[2572437])
print(sourceProb[2572438])
print(target2Prob[0])


target1FilePath = os.path.join(os.path.dirname(__file__), target1Path)
target2FilePath = os.path.join(os.path.dirname(__file__), target2Path)


targetF1 = open( target1FilePath, 'w' )
for i in target1Prob:
	targetF1.write(i + '\n')
targetF1.close()


targetF2 = open( target2FilePath, 'w' )
for i in target2Prob:
	targetF2.write(i + '\n')
targetF2.close()