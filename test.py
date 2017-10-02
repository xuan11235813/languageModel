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


#------------------------read-IBM-file-test---------------------------------------------------

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

#np.save("data/weight", a)

b = np.load('data/lexicon_weight_projection.npy')


print(a)
print(len(b))
print(len(b[0]))

testFunction()
testFunction(1)

#------------------------------string compare test---------------------------------------


class testCase:
	def __init__ (self, mode = ''):
		if mode == 'lstm':
			self.a = 'a'
			print('hello lstm')
		elif mode == 'rnn':
			self.b = 'b'
			print('hello rnn')
		else:
			print('hello')
			self.c = 'c'
	def helloa(self):
		print(self.a)
	def hellob(self):
		print(self.b)
	def helloc(self):
		print(self.c)
#-------------------------------concat-stack-select-test------------------------------------

projection = tf.Variable(tf.random_normal([500,200,10]))
print(projection.get_shape())

init = tf.global_variables_initializer();

sess = tf.Session()       
        #initialize
sess.run(init)

print(sess.run(projection[1,:]))

print(len(sess.run(projection[1,:])))	
print(len(sess.run(projection[1,:,2])))
print(sess.run(projection[:,2,:]))

result = []

for i in range(3):
	result.append(projection[1,1,:])

tfResult = tf.stack(result)

tfFinal = tf.concat(result, 0)

tfSelect = [result[i] for i in [1,2]]
tfFinalSelect = tf.concat(tfSelect, 0)
print(tfFinalSelect.get_shape())
print(sess.run(tfResult))
print(sess.run(tfFinal))
print(sess.run(tfFinalSelect))
print(sess.run(tf.add(tfSelect[0], tfSelect[1])))


#-------------------------condition test-------------------------------------
sess = tf.Session()

a = tf.Variable([[3,3],[3,3]])

def if_true():
	return tf.matmul(a,a)

def if_false():
	return tf.add(a,a)

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
init = tf.global_variables_initializer();

result = tf.cond( tf.less(x,y), if_true, if_false)
#result = tf.cond(tf.less(x,y), lambda: tf.matmul(a,a), lambda: tf.add(a,a))
sess.run(init)
print(sess.run(a))
p = sess.run([result], feed_dict={x:3, y:4} )
print(p)
'''

#-----------------------while-loop-test-----------------------------------

sess = tf.Session()
i0 = tf.constant(0)
m0 = tf.ones([2,2])
i_max = tf.constant(10)

c = lambda i, m: i<10
b = lambda i, m: [i +1, tf.concat([m,m], 0)]
def cond(i,m, iMax):
	return tf.less(i, iMax)

def body(i,m, iMax):
	i = tf.add(i,1)
	m = tf.concat([m,m], 0)
	return i ,m, iMax

r = tf.while_loop(
	cond,body, 
	loop_vars=[i0, m0, i_max], 
	shape_invariants = 
	[i0.get_shape(), 
	tf.TensorShape([None, 2]), 
	i_max.get_shape()])

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(r)[0])
print(len(sess.run(r)[1]))
print(sess.run(r)[2])

