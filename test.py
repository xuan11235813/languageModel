import tensorflow as tf


x = [1,3,1]

a = tf.Variable(tf.random_normal([5, 4]))

b = tf.Variable([], dtype = tf.float32)

for word in x:
	b = tf.concat([b,tf.gather(a,word)],0)
#b1 = tf.gather(a, 0)
#b2 = tf.gather(a, 2)
#b3 = tf.gather(a, 3)
#c = tf.concat([b1,b2,b3],0)
init = tf.global_variables_initializer();

sess = tf.Session()
sess.run(init)
print(sess.run(a))
print(sess.run(b))
#print(sess.run(c))