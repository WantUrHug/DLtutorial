import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
#把自己定义的一些函数import进来
import input_data
import model

#预定义一些参数，用全大写的方式
N_CLASSES = 2
IMAGE_W = 160
IMAGE_H = 160
CHANNELS = 3
BATCH_SIZE = 20
CAPACITY = 2000
MAX_STEP = 5000
learning_rate = 0.0001

#该文件夹中包含了1000张猫的图片和1000张狗的图片
train_dir = "D:\\Download\\DogCat\\train\\all\\"
logs_train_dir = "C:\\Users\\60214\\Desktop\\python_work\\DeepLearning\\Dog_Cat\\log\\"

file_image, file_label = input_data.get_files(train_dir)

#原本想着把生成器写在 input_data.py 中然后再 import 进来，但是有一个问题，就是 tf.data.Dataset.from_generator
#获取一个生成器函数，似乎是获取函数名，然后我看了网上一些简单例子(没看github)，生成器都是方便使用没有形式参数，而我
#这里用默认参数的形式来做，应该也没问题
def data_gen(image_list = file_image, label_list = file_label, image_W = IMAGE_W, image_H = IMAGE_H, shuffle = True):

	#先把数组转化成np.ndarray对象，方便打乱顺序
	image_list = np.array(image_list)
	label_list = np.array(label_list)

	if shuffle:
		ids = list(np.arange(len(image_list)))
		image_list = image_list[ids]
		label_list = label_list[ids]

	#按照batch-size来生成批次数据，先计算一共有25000张图片
	#每个批次如果是默认的100，那么一共有250个batch
	#注意range()中必须是整数，所以加上int类型转换
	for i in range(len(image_list)):
		
		#空数组准备，每次都需要初始化这个
		image_data = np.zeros((image_H, image_W, 3), dtype = "float")		

		#确定是第几张图片，使用Image.open读取
		im = Image.open(image_list[i])
		#修改统一尺寸
		im = im.resize((image_H, image_W))
			
		#因为图片都是像素值从0到255，这么大的输入范围对模型的训练不好，所以，
		#要进行缩放，我选择的是直接除以255，也就是缩放到了0-1的范围内。
		im_array = np.array(im)/255
		image_data = im_array

		#这个地方不对输出的标签进行处理，是因为，后面要给到计算交叉熵的函数
		#tf.nn.sparse_softmax_cross_entropy_with_logits中，这个函数需
		#要接受一个logits表示模型输出的结果和一个lables作为正确的结果，
		#logits是一个一维张量，而labels是零维张量，当然加上batch之后logits
		#是二维而labels是一维，总之他们维度的差距是1
		yield image_data, label_list[i]

#tensorflow 推荐不使用队列的形式来feed数据，而使用Dataset少量数据
#用split_from_tensor，直接从内存中读取张量，较大的数据集的话得使用
#from_generator，自己定义生成器
train_dataset = tf.data.Dataset.from_generator(
	generator = data_gen,
	#输出的是图像像素数据和标签，自然一个是float32一个是int32
	output_types = (tf.float32, tf.int32))

#再次打乱、规定每次输出的batch_size，然后repeat是可以循环，跑完所有的数据之后从头
#开始。remainder参数表示当跑到全部数据的最后时可能取不整，也就是剩下的数据数量不足
#一个batch_size，是继续(True)输出还是丢弃(False)。
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder = False).repeat()

#dataset实际也是一个生成器，所以要生成一个iterator来迭代获取其中的数据
train_iterator = tf.data.make_one_shot_iterator(train_dataset)

train_handle = tf.placeholder(tf.string, shape = [])
iterator_from_train_handle = tf.data.Iterator.from_string_handle(
	train_handle, train_dataset.output_types, train_dataset.output_shapes)
#关键一步，从迭代器中生成下一批数据
next_train_data, next_train_label = iterator_from_train_handle.get_next()

#定义占位符
x = tf.placeholder(tf.float32, shape = [None, IMAGE_H, IMAGE_W, CHANNELS])
#计算前向传播
y_pred = model.cnn_inference(inputs = x, num_classes = N_CLASSES)

y_true = tf.placeholder(tf.int64, shape = [None])
train_loss = model.losses(logits = y_pred, labels = y_true)
train_accuracy = model.evaluation(logits = y_pred, labels = y_true)
train_op = model.training(train_loss, learning_rate = 0.0001)

#summary_op = tf.summary.merge_all()

init_op = tf.global_variables_initializer()

#定义一个字典，来储存过程中的损失值和准确率
history = {}
history["train_loss"] = []
history["train_acc"] = []

with tf.Session() as sess:

	sess.run(init_op)
	training_handle = sess.run(train_iterator.string_handle())

	#train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

	saver = tf.train.Saver()

	#如何衡量训练过程的准确率和损失呢？因为每一个step都是换了一批数据，所以如果每个step都计算一次，
	#那么最后得到的结果肯定是可以看出进步的，但是过程必然非常崎岖。过于崎岖变化的训练过程对我们没有
	#太大的参考价值，所以有两个方法，第一就是每隔50步测一个batch中的损失和正确率，第二就是把每50步
	#测一次平均值，此处考虑第二种，所以用以下两个变量来储存，记得每50步要置零一次
	sum_train_loss = 0
	sum_train_acc = 0

	#step从1开始，然后最大是到MAX_STEP，也是一共MAX_STEP，然后避免了开始有一个step
	#为0对于判断的尴尬，而且后来在输出数组时也方便去绘图
	for step in range(1, MAX_STEP + 1):

		#使用句柄从迭代器中获取数据，再feed到训练中去
		data_value, label_value = sess.run((next_train_data, next_train_label), feed_dict = {train_handle: training_handle})
		_, loss_value, acc_value = sess.run((train_op, train_loss, train_accuracy), feed_dict = {x: data_value, y_true: label_value})


		#每50步要记录一下
		if step % 50 == 0:
			history["train_loss"].append(sum_train_loss/50)
			history["train_acc"].append(sum_train_acc/50)
			print("Step %d, lastest train loss = %.2f and train accuracy = %.2f%%" % (step, history["train_loss"][-1], history["train_acc"][-1] * 100.0))
			sum_train_loss = 0
			sum_train_acc = 0
			#summary_str = sess.run(summary_op)
			#train_writer.add_summary(summary_str, step)			
		else:
			sum_train_loss += loss_value
			sum_train_acc += acc_value

		if step%5000 == 0 or step == MAX_STEP:
			#保存模型，保存 Session 中的参数
			saver.save(sess, logs_train_dir + "my_model", global_step = step)

STEPS = range(1, MAX_STEP + 1, 50)

#plt.subplot(1,2,1)
plt.plot(STEPS, history["train_loss"], "b", label = "train loss")
plt.plot(STEPS, history["train_acc"], "r", label = "train accuracy")
plt.legend()

plt.savefig(logs_train_dir + "1.jpg")
plt.show()