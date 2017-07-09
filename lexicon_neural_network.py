import tensorflow as tf

training_epochs = 15
batch_size = 128

n_input = 8

#projection layer
n_projection = 40680
n_projection_output = 100
#1st hidden layer
n_hidden_1 = 800

#2nd hidden layer
n_hidden_2 = 500

#output class layer
n_classes = 2000

weights = {
    'projection': tf.Variable(tf.random_normal([n_projection, n_projection_output])),
    'hidden': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'bHidden': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def multilayer_perceptron(sourceTarget, weights, biases):

    b = tf.Variable([], dtype = tf.float32)
    for word in sourceTarget:
        b = tf.concat([b,tf.gather(weights['projection'],word)],0)

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(b, weights['hidden']), biases['bHidden'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

    return out_layer



x = tf.placeholder("int", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-1 * cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))