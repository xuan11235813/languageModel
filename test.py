import tensorflow as tf
import numpy as np
import math as mt
import para

'''
x = [1,3,1]

a1 = tf.Variable(tf.random_normal([5, 4]))
a2 = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable([], dtype = tf.float32)
h = tf.Variable([], dtype = tf.float32)
y = tf.Variable([2,2,2,2,2,2,4])
y2 = tf.Variable([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
y1 = tf.Variable([[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
h1 = tf.placeholder(tf.float32, [None, 12])


for word in x:
	b = tf.concat([b,tf.gather(a1,word)],0)

p = tf.unstack(y,None,0)
hp = tf.gather(a1,y)
h = tf.concat(tf.unstack(tf.gather(a1,y),None,0),0)
s = []
for i in range(10):
	s.append(h)
#b1 = tf.gather(a, 0)
#b2 = tf.gather(a, 2)
#b3 = tf.gather(a, 3)
#c = tf.concat([b1,b2,b3],0)
init = tf.global_variables_initializer();

sess = tf.Session()
sess.run(init)
print(sess.run(a1))
print(sess.run(b))
_a1 = sess.run(a1)
_a2 = sess.run(a2)
print(a1)
print(np.dot(_a1,_a2))


array1 = np.ones(5,dtype = float) * np.array([1.3,1.2,1.5,1.6,1.8])
print(array1)
print(array1[1] / array1[3])
print(mt.fsum(array1))
#print(sess.run(c))

weight = {}
weight['fuck'] = 8
print(weight['fuck'])


print(sess.run(y))
print(sess.run(hp))
print(sess.run(h))


for i in range(10):
	print("Am I stupid?")


def fuck(i):
	return i*3, i*4

a,b = fuck(5)
print(a)
print(b)


for i in range(10):
	print(sess.run(s[i]))



print(sess.run(y1))

matrix = []
for i in range(5):
	matrix.append(y1)

matrix = tf.concat(matrix, 0)

a = tf.reshape(y2, [5,1,4])
matrix = tf.reshape(matrix, [5,4,3])
x = tf.matmul(a, matrix)
print(sess.run(x))
print(sess.run(tf.reshape(x, [5,3])))
print(sess.run(tf.nn.softmax(tf.to_float(matrix))))


IBMDic = {}

para = para.Para()
filePath = para.GetIBMFilePath()

try:
	file = open(filePath, "r")
	for line in file:
		item = []
		for word in line.split(" "):
			item.append(word)
		itemDic = {}
		itemDic[item[1].rstrip()] = float(item[2].rstrip())
		if item[0].rstrip() in IBMDic:
			IBMDic[item[0].rstrip()][item[1].rstrip()] = float(item[2].rstrip())
		else:
			IBMDic[item[0].rstrip()] = itemDic


except IOError as err:
	print("target vocabulary files do not exist")
	self.alert += 1
'''
def testFunction(s = 0):
	if s == 0:
		print('hey baby')
	else:
		print('yeah')

weight = tf.Variable(tf.random_normal([3,5]))
sess = tf.Session()
init = tf.global_variables_initializer();

sess.run(init)

a = sess.run(weight)

np.save("data/weight", a)

b = np.load('data/weight.npy')


print(a)
print(b)

testFunction()
testFunction(1)


