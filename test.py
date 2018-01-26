import tensorflow as tf
import numpy as np
import math as mt
import para
'''
x = [1,3,1]
i = 0
a1 = tf.Variable(tf.random_normal([5, 4]))
a2 = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable([], dtype = tf.float32)
h = tf.Variable([], dtype = tf.float32)
y = tf.Variable([2,2,2,2,2,2,4])
y2 = tf.Variable([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
y1 = tf.Variable([[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
y3 = tf.Variable([[2,2],[2,5]])
if (i == 1):
	y4 = tf.Variable([[2,2],[2,5]])
else:
	y4 = tf.Variable(tf.random_normal([2, 2]))

y5 = tf.matmul(y3, y4);
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

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
print(sess.run(y5))

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
print(sess.run(matrix))

a = tf.reshape(y2, [5,1,4])
matrix = tf.reshape(matrix, [5,4,3])
print(sess.run(matrix))

#x = tf.matmul(a, matrix)
#print(sess.run(x))
#print(sess.run(tf.reshape(x, [5,3])))

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

def if_false():return tf.add(a,a)
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
init = tf.global_variables_initializer();

#result = tf.cond(tf.less(x,y), lambda: tf.matmul(a,a), lambda: tf.add(a,a))
result = tf.cond( tf.less(x,y), if_true, if_false)
sess.run(init)
print(sess.run(a))
p = sess.run([result], feed_dict={x:3, y:4} )
print(p)


#-----------------------basic-while-loop-test-----------------------------------

sess = tf.Session()
i0 = tf.constant(0)
m0 = tf.ones([2,2])
toAdd0 = tf.ones([1,2])
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



#-----------------advanced-while-loop-test------------------------

sess = tf.Session()
i0 = tf.constant(0)
i_Max = tf.constant(10)
j_Max = tf.constant(10)
m0 = tf.ones([0,2])
mAdd0 = tf.ones([1,2])
mp = tf.ones([200,2])
s = 1

def bodySource(j, jMax, m, mAdd):
	j = tf.add(j, 1)
	mAdd = tf.add(mAdd, s)
	mAppendix = mp[tf.subtract(j,1), :]
	m = tf.concat([m, mAdd, [mAppendix]], 0)
	return j, jMax, m, mAdd

def condSource(j, jMax, m, mAdd):
	return tf.less(j, jMax)


def condTarget(i, iMax, jMax, m, mAdd):
	return tf.less(i, iMax)

def bodyTarget(i, iMax, jMax, m, mAdd):

	j0 = tf.constant(0)
	sourceLoop = tf.while_loop(
		condSource, 
		bodySource,
		loop_vars = [j0, jMax, m, mAdd],
		shape_invariants = [
		j0.get_shape(),
		jMax.get_shape(),
		tf.TensorShape([None, 2]),
		mAdd.get_shape()])
	i = tf.add(i,1)
	m = sourceLoop[2]
	return i, iMax, jMax, m, mAdd 

targetLoop = tf.while_loop(
	condTarget,
	bodyTarget,
	loop_vars = [i0, i_Max, j_Max, m0, mAdd0],
	shape_invariants = [
	i0.get_shape(),
	i_Max.get_shape(),
	j_Max.get_shape(),
	tf.TensorShape([None, 2]),
	mAdd0.get_shape()])

print(sess.run(targetLoop)[3])


y1 = tf.Variable([[2,2,2],[3,3,3],[4,4,4],[5,5,5]])

n1 = tf.Variable(2)
n2 = tf.Variable(2)

y2, y3 = tf.split(y1, [n1,n2], 0)

sess = tf.Session()

init = tf.global_variables_initializer();
sess.run(init)
print(sess.run(y2))
print(sess.run(y3))


#--------------------save-and-restore-test------------------------
M = tf.Variable(tf.random_normal([2,2]),name="M")

x = tf.Variable([[2.0],[3.0]], name="x",dtype=tf.float32)

y = tf.matmul(M,x)

sess = tf.Session()

saver = tf.train.Saver({
	"M" : M,
	"x":x
	})

init = tf.global_variables_initializer()

sess.run(init)
l = sess.run(y)
print(l)

saver.save(sess, 'a.ckpt')


'''

M = tf.Variable(tf.random_normal([2,2]), name="M")

x = tf.Variable([[2.0],[3.0]], name="x", dtype=tf.float32)

y = tf.matmul(M,x)

sess = tf.Session()

saver = tf.train.Saver()

init = tf.global_variables_initializer()

sess.run(init)

saver.restore(sess, 'a.ckpt')

l = sess.run(y)
print(l)

