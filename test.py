import tensorflow as tf
import numpy as np
import math as mt

x = [1,3,1]

a1 = tf.Variable(tf.random_normal([5, 4]))
a2 = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable([], dtype = tf.float32)

for word in x:
	b = tf.concat([b,tf.gather(a1,word)],0)
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