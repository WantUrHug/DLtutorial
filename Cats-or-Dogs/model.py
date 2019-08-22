import tensorflow as tf

def cnn_inference(inputs, num_classes = 2):
	
	#用底层的方式来写代码
	with tf.variable_scope("conv1") as scope:
		#第一层的卷积层conv1，卷积核为3x3x3，有16个
		weights = tf.get_variable(
			"weights",
			shape = [3, 3, 3, 16],
			dtype = tf.float32,
			initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))
		#偏置，每一个卷积核对应一个偏置值
		biases = tf.get_variable(
			"biases",
			shape = [16],
			dtype = tf.float32,
			initializer = tf.constant_initializer(0.1))
		#卷积时的步长，[1, x_movement, y_movement, 1]，并选择全0填充
		conv = tf.nn.conv2d(inputs, weights, strides = [1,1,1,1], padding = "SAME")
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name = scope.name)

	#最大池化层和LRN层组合
	with tf.variable_scope("pooling1_lrn") as scope:
		#池化的尺寸是2x2，步长是[2,2]
		#有一个要注意，池化层的padding和卷积层的padding含义不同，卷积层的padding是卷积运算后
		#尺寸维持不变，而池化层的padding，肯定是不能保持不变的，不然就是失去了subsample降采样
		#的意义。所以池化层的padding为SAME，指如果特征的尺寸和池化层的kernel不能整除时，使用0
		#填充使得得以完整，而VALID表示抛弃那边缘的部分。而我们这里特征的尺寸都是偶数而卷积核也是
		#2x2所以padding为SAME或者VALID都是一样的
		pool1 = tf.nn.max_pool(
			conv1,
			ksize = [1, 2, 2, 1], 
			strides = [1, 2, 2, 1],
			padding = "SAME",
			name = "pooling1")
		#Local response normalization，局部相应归一化
		norm1 = tf.nn.lrn(
			pool1,
			depth_radius = 4,
			bias = 1.0,
			alpha = 0.001/9.0,
			beta = 0.75,
			name = "normal1")

	#第二个卷积层，也是16个卷积核，只是每个卷积核，为了前面的对应，所以形状是[3,3,16]
	with tf.variable_scope("conv2") as scope:
		weights = tf.get_variable('weights',
                                  shape = [3, 3, 16, 16],  #这里只有第三位数字16需要等于上一层的tensor维度
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
		biases = tf.get_variable('biases',
                                 shape = [16],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1)) 
		conv = tf.nn.conv2d(norm1, weights, strides = [1, 1, 1, 1], padding = "SAME")
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name='conv2')

	#第二个池化层和LRN层，这里把池化放在LRN后面
	with tf.variable_scope("pooling2_lrn") as scope:
		norm2 = tf.nn.lrn(
			conv2,
			depth_radius = 4,
			bias = 1.0,
			alpha = 0.001/9.0,
			beta = 0.75,
			name = "normal2")
		pool2 = tf.nn.max_pool(
			norm2,
			ksize = [1, 2, 2, 1],
			strides = [1, 2, 2, 1],
			padding = "SAME",
			name = "pooling2")

	#第三层为全连接层
	with tf.variable_scope("local3") as scope:
		#把原本的四维张量展开成二维的，保持每个batch中的数据仍在原来的
		#batch中，应该可以用 Flatten 实现，但不知道这个batch
		#能否维持，尚待了解
		reshape = tf.reshape(pool2, shape = [-1, 25600])

		#在定义参数矩阵时，必须要给定shape，不能有未知量在其中
		weights = tf.get_variable(
			"weights",
			shape = [25600, 256],
			dtype = tf.float32,
			initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))

		biases = tf.get_variable(
			"biases",
			shape = [256],
			dtype = tf.float32,
			initializer = tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name = scope.name)

	#第四层为全连接层
	with tf.variable_scope("local4") as scope:

		weights = tf.get_variable(
			"weights",
			shape = [256, 512],
			dtype = tf.float32,
			initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))

		biases = tf.get_variable(
			"biases",
			shape = [512],
			dtype = tf.float32,
			initializer = tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name = "local4")

	#第五层是softmax处理前的一个全连接层，只是没有加上softmax激活函数
	with tf.variable_scope("softmax_linear") as scope:

		weights = tf.get_variable(
			"weights",
			shape = [512, num_classes],
			dtype = tf.float32,
			initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))

		biases = tf.get_variable(
			"biases",
			shape = [num_classes],
			dtype = tf.float32,
			initializer = tf.constant_initializer(0.1))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name = "softmax_linear")
		#这里只是命名为softmax_linear，真正的softmax函数放在下面的losses函数里面和交叉熵结合在一起了，这样可以提高运算速度

	return softmax_linear

def losses(logits, labels):
	'''
	定义损失函数
	'''
	with tf.variable_scope("loss") as scope:
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = "loss_per_eg")
		loss = tf.reduce_mean(cross_entropy, name = "loss")

	return loss

def training(loss, learning_rate):
	'''
	确给定学习率，指定训练时使用的优化器。
	添加一个不可训练的参数 global_step，来方便后续一些保存模型之类的工作
	'''

	with tf.variable_scope("optimizer"):
		#选择Adam优化器
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		global_step = tf.Variable(0, name = "global_step", trainable = False)
		train_op = optimizer.minimize(loss, global_step = global_step)

	return train_op

def evaluation(logits, labels):
	'''
	返回正确率。
	'''

	with tf.variable_scope("accuracy") as scope:
		#之前模型的输出还没有经过softmax层处理，但实际上也可以不用加softmax层，因为最大的
		#列经过处理还是最大的
		prediction = tf.nn.softmax(logits)
		#tf.argmax返回一个张量中，某一行中最大那个数值的位置，例如
		#[[0.1,0.9],[0.6,0.4]]，第二个参数1指的是第二维(从0开始)，所以返回的是[1,0]，表示在每一行中
		#最大的数值所在的位置分别是1和0
		correct = tf.equal(tf.argmax(logits, 1), labels)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	return accuracy