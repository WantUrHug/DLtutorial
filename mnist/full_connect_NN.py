import tensorflow as tf
import numpy as np

NUM_CLASSES = 10
BATCH_SIZE = 128
MAX_STEPS = 5000
CHECK_STEPS = 100

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data("mnist.npz")

train_y = tf.one_hot(train_y, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder = True).repeat()
iterator = train_dataset.make_one_shot_iterator()
next_data, next_label = iterator.get_next()


X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32,[None, NUM_CLASSES])

w1 = tf.Variable(tf.truncated_normal([28*28, 16], dtype = tf.float32, mean=0, stddev=.01))
b1 = tf.Variable(tf.truncated_normal([16], dtype = tf.float32, mean=0, stddev=.01))

outputs = tf.matmul(X, w1) + b1

w2 = tf.Variable(tf.truncated_normal([16, 32], dtype = tf.float32, mean=0, stddev=.01))
b2 = tf.Variable(tf.truncated_normal([32], dtype = tf.float32, mean=0, stddev=.01))

outputs = tf.matmul(outputs, w2) + b2

w3 = tf.Variable(tf.truncated_normal([32, 64], dtype = tf.float32, mean=0, stddev=.01))
b3 = tf.Variable(tf.truncated_normal([64], dtype = tf.float32, mean=0, stddev=.01))

outputs = tf.matmul(outputs, w3) + b3

w4 = tf.Variable(tf.truncated_normal([64, NUM_CLASSES], dtype = tf.float32, mean=0, stddev=.01))
b4 = tf.Variable(tf.truncated_normal([NUM_CLASSES], dtype = tf.float32, mean=0, stddev=.01))

outputs = tf.matmul(outputs, w4) + b4

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = outputs, labels = Y))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs, 1), tf.argmax(Y, 1)), tf.float32))

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	for i in range(1,1+MAX_STEPS):

		data_value, label_value = sess.run((next_data, next_label))
		data_value = data_value.reshape(BATCH_SIZE, 28*28)
		sess.run(train_op, feed_dict = {X: data_value, Y: label_value})

		if i%CHECK_STEPS == 0:
			loss_value, acc_value = sess.run((loss, accuracy), feed_dict = {X: data_value, Y: label_value})
			print("Step %d, loss = %.2f, accuracy = %.4f." % (i, loss_value, acc_value))

	test_data = test_x[:500].reshape(500, 28*28)
	test_label = tf.one_hot(test_y[:500], depth = 10)
	test_label = sess.run(test_label)

	print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y:test_label}))