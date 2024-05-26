import cv2
import numpy as np
import NLDF
import os
import sys
import tensorflow as tf
import time
import vgg16
#https://stackoverflow.com/questions/41607144/loading-two-models-from-saver-in-the-same-tensorflow-session
#https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/


save_dir = 'result'
#--------------------------------------------------------------

def sliPreProcess(img_size):
	img = cv2.imread(sys.argv[1])
	img_shape = img.shape
	img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
	img = img.reshape((1, img_size, img_size, 3))
	return img,img_shape

def sliPostProcess(result,label_size,img_shape):
	result = np.reshape(result, (label_size, label_size, 2))
	result = result[:, :, 0]
	result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))
	f = sys.argv[1]
	fn = f.split('.')
	save_name = os.path.join(save_dir, fn[0]+'_Salient.png')       
	cv2.imwrite(save_name, (result*255).astype(np.uint8))
	
model = NLDF.Model()
model.build_model()

sli_sess = tf.Session(graph=model)
sli_sess.run(tf.global_variables_initializer())
img_size = NLDF.img_size
label_size = NLDF.label_size
ckpt = tf.train.get_checkpoint_state('Model/')
saver = tf.train.Saver()
saver.restore(sli_sess, ckpt.model_checkpoint_path)

img,img_shape = sliPreProcess(img_size)
result = sli_sess.run(model.Prob,feed_dict={model.input_holder: img})
sliPostProcess(result,label_size,img_shape)
#-----------------------------------------------------
