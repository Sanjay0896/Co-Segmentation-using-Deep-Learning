import tensorflow as tf
import numpy as np
from model_triplet import *
import cv2
import os
import sys
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_size = 480
mean = 1.50

lst = sys.argv[1].split('/')
image_name1 = lst[len(lst)-1].split('.')[0]
lst = sys.argv[2].split('/')
image_name2 = lst[len(lst)-1].split('.')[0]


left_img = './result/'+image_name1+'_crop.jpg'
right_img = './result/'+image_name2+'_crop.jpg'


R=255
G=255
B=255

def extract(input_original,input_salient,output):
	img1 = cv2.imread(input_original)
	img11 = cv2.resize(img1,(image_size,image_size))
	img2 = cv2.imread(input_salient)
	img21 = cv2.resize(img2,(image_size,image_size))
	img31 = np.copy(img11)

	for i in range(0,image_size):
		for j in range(0,image_size):
			flag = False
			for k in range(0,3):
				if img21[i,j,k] != 0:
					flag = True
					break
			if flag == False:
				img31[i,j,0] = R
				img31[i,j,1] = G
				img31[i,j,2] = B
		
	cv2.imwrite(output,img31)
	

def euclidean_distance(x, y):
    d = tf.square(tf.subtract(x, y))
    return tf.sqrt(tf.reduce_sum(d,1))
	
'''def euclidean_distance(x, y):
    a = tf.square(tf.subtract(x, y))
	b = tf.reduce_sum(a,1)
	if b.eval() == 0:
		return tf.constant(0)
	else
		return tf.sqrt(b)'''
	
	
img_placeholder = tf.placeholder(tf.float32, [None,image_size,image_size, 3], name='img')
net = mynet(img_placeholder, reuse=False)

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,"m3/m.ckpt")
	img1 = cv2.imread(left_img)
	print left_img	
	print img1
	img1 = np.divide(img1,255.0)
	img2 = cv2.imread(right_img)
	img2 = np.divide(img2,255.0)
	img1 = cv2.resize(img1,(image_size,image_size))
	img2 = cv2.resize(img2,(image_size,image_size))
	left = sess.run(net, feed_dict={img_placeholder:[img1]})
	right = sess.run(net, feed_dict={img_placeholder:[img2]})
	dist = euclidean_distance(left,right)
	d=dist.eval()
	print d
	if d>=mean:
		d = 0.0
	else:
		d = 1.0
	
	print('output:',d)
	
	if d==1:
		extract(sys.argv[1],'./result/'+image_name1+'_Salient.png','coSeg1.jpg')
		extract(sys.argv[2],'./result/'+image_name2+'_Salient.png','coSeg2.jpg')
		image2 = mpimg.imread("coSeg1.jpg")
		plt.imshow(image2)
		plt.show()		
		image3 = mpimg.imread("coSeg2.jpg")
		plt.imshow(image3)
		plt.show()		
		
	else:
		print("Images are different")

