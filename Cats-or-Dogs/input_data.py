#encoding = utf-8
import tensorflow as tf
import numpy as np
import os
#from sklearn import preprocessing


def get_files(file_dir):
	'''
	处理所有图片，包括猫和狗，都在同一个文件夹下的情况
	'''

	#定义空的数组，来储存图片的绝对路径
	cats = []
	label_cats = []
	dogs = []
	label_dogs = []

	#历遍整个文件夹，因为存放图片的文件夹中，图片的格式都是“cat.1.jpg”，“dog.1.jpg”的形式，
	#所以使用split方法，分辨以下图片是猫还是狗，然后把文件名加上文件夹的名称形成路径信息，存
	#放到对应的数组中
	for file in os.listdir(file_dir):
		name = file.split(sep = ".")
		if name[0] == "cat":
			cats.append(file_dir + file)
			label_cats.append(0)
		else:
			dogs.append(file_dir +file)
			label_dogs.append(1)

	#看一下各有多少张图片
	print("There are %d cats\nThere are %d dogs." % (len(cats), len(dogs)))

	#np.hstack把两个一维的向量水平相连，仍为一维(25000,)
	image_list = np.hstack((cats, dogs))
	label_list = np.hstack((label_cats, label_dogs))

	#组合后的shape为(2,25000)，第一行为图片路径信息，第二行为标签信息
	tmp = np.array([image_list, label_list])
	#进行转置，shape为(25000,2)，第一列为图片路径信息，第二列为标签信息
	tmp = tmp.transpose()
	#打乱
	np.random.shuffle(tmp)

	#把打乱后的数组，取出第一列作为即将输出的图片路径信息，第二列则是标签信息
	image_list = list(tmp[:, 0])
	label_list = list(tmp[:, 1])
	label_list = [int(float(i)) for i in label_list]

	return image_list, label_list


if __name__ == "__main__":

	train_dir = "D:\\Download\\train\\"
	i, l = get_files(train_dir)



