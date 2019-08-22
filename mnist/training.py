import tensorflow as tf
#导入自己定义的前向传播模型
import model

#定义一些参数
BACTH_SIZE = 128
TIMESTEPS = 28
INPUTS = 28
NUM_HIDDEN = 150
NUM_CLASSES = 10
MAX_STEPS = 3000
LEARNING_RATE = 1e-4

#导入数据
(train_x, train_y),(test_x, test_y) = tf.keras.datasets.mnist.load_data(path = "mnist.npz")
print(train_x.shape)
print(train_y.shape)
train_y = tf.one_hot(train_y, depth = 10)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.shuffle(1000).batch(BACTH_SIZE, drop_remainder = True).repeat()

train_iterator = tf.data.make_one_shot_iterator(train_dataset)

train_handle = tf.placeholder(tf.string, [])
iterator_from_train_hanlde = tf.data.Iterator.from_string_handle(
	train_handle, train_dataset.output_types, train_dataset.output_shapes)

next_train_data, next_train_labels = iterator_from_train_hanlde.get_next()

X = tf.placeholder(tf.float32, shape = [BACTH_SIZE, TIMESTEPS, INPUTS])
Y = tf.placeholder(tf.float32, shape = [BACTH_SIZE, NUM_CLASSES])

y = model.RNN_v1(X, num_hidden = NUM_HIDDEN, num_classes = NUM_CLASSES)

loss = model.loss(logits = y, labels = Y)
train_op = model.training(loss = loss, learning_rate = LEARNING_RATE)
accuracy = model.evaluation(logits = y, labels = Y)

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	training_handle = sess.run(train_iterator.string_handle())

	sum_train_loss = 0
	sum_train_acc = 0

	for step in range(1, MAX_STEPS + 1):

		data_value, label_value = sess.run((next_train_data, next_train_labels), feed_dict = {train_handle: training_handle})
		_, loss_value, accuracy_value = sess.run((train_op, loss, accuracy), feed_dict = {X: data_value, Y: label_value})

		sum_train_loss += loss_value
		sum_train_acc += accuracy_value

		if step%100 == 0:			
			print("Step %d, loss = %.2f, accuracy = %.2f%%."%(step, sum_train_loss/100, sum_train_acc))
			sum_train_loss = 0
			sum_train_acc = 0


