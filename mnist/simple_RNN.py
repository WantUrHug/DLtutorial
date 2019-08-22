import tensorflow as tf

#定义模拟过程的变量
BATCH_SIZE = 128
TIMESTEP = 28
INPUT_DIM = 28
NUM_CLASSES = 10
NUM_HIDDEN = 100
MAX_STEPS = 3000
CHECHK_STEPS = 150


X = tf.placeholder(tf.float32, [None, TIMESTEP, INPUT_DIM])
Y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

#创建一个BasicRNN单元
cell = tf.nn.rnn_cell.BasicRNNCell(NUM_HIDDEN)
#dynamic_nn，执行批次输入与BasicRNN单元循环运算的关键
outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
#取出batch-size数据中的最后输出，state状态丢弃，用_表示
outputs = outputs[:, -1, :]

#计算完RNN的结果之后，由于NUM_HIDDEN和最重要输出的向量维度不同，所以添加一个没有激活函数的
#全连接网络来调整
w1 = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], dtype = tf.float32, mean = 0, stddev = 0.1))
b1 = tf.Variable(tf.truncated_normal([NUM_CLASSES], dtype= tf.float32, mean = 0, stddev = 0.1))

final = tf.matmul(outputs, w1) + b1

#定义损失函数和训练的方式
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = final, labels = Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
#计算准确率
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final, 1), tf.argmax(Y, 1)), tf.float32))

#获取数据
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data("mnist.npz")
#从数据集中拿到的标签都是一个数字，从0到9，所以用tf.ont_hot转化成one-hot向量
train_y = tf.one_hot(train_y, depth = 10)

#因为要使用mini-batch训练，所以使用tensorflow推荐的tf.data来导入数据
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE).repeat()
train_iter = train_dataset.make_one_shot_iterator()

(next_data, next_label) = train_iter.get_next()

#准备好测试用的数据
test_data = test_x[:500]/255.
test_label = test_y[:500]

with tf.Session() as sess:

	#全局初始化
	sess.run(tf.global_variables_initializer())

	for step in range(1, MAX_STEPS + 1):
	
		data, label = sess.run((next_data, next_label))
		#非常关键的一步！
		#一开始忘了要把像素值缩放到0-1的范围内，导致模型优化的正确率始终只能到
		#60%左右，记住，最重要的步骤往往都是数据的处理
		data = data/255.0
		
		sess.run(train_op, feed_dict = {X: data, Y: label})

		if step%CHECHK_STEPS == 0:
			loss_value, acc_value = sess.run((loss, accuracy_op), feed_dict = {X: data, Y: label})
			print("Step %d, loss = %.2f, accuracy = %.2f%%"%(step, loss_value, acc_value*100))

	#测试数据的标签也需要转化成one-hot向量
	test_label = sess.run(tf.one_hot(test_label, depth = 10))
	print("Accuracy in test data is: %.2f%%" % (sess.run(accuracy_op, feed_dict = {X: test_data, Y: test_label})*100))