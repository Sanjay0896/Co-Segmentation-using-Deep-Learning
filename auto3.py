import tensorflow as tf
import numpy as np
from model_triplet import *
import cv2
import sys
#from tensorflow.python import pywrap_tensorflow
import NLDF
import tensorflow as tf
import time
import vgg16

save_dir = 'result'
image_size = 480
mean = 1.3523
left_img = sys.argv[1]
right_img = sys.argv[2]


#----------------------------------------------------------------

'''def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    varlist=[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
    return varlist
varlist=print_tensors_in_checkpoint_file(file_name='m3/m.ckpt',all_tensors=True,tensor_name=None)
#variables = tf.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
#saver = tf.train.Saver(variable[:len(varlist)])
print('-----')
print(varlist)
#print(variables)
#print(variables[:len(varlist)])
print('-----')'''
#----------------------------------------------------------------
def simPreProcess(img_path):
	img = cv2.imread(img_path)
	img = np.divide(img,255.0)
	img = cv2.resize(img,(image_size,image_size))
	return img

def euclidean_distance(x, y):
    d = tf.square(tf.subtract(x, y))
    return tf.sqrt(tf.reduce_sum(d,1))
	
img_placeholder = tf.placeholder(tf.float32, [None,image_size,image_size, 3], name='img')
#net = mynet(img_placeholder, reuse=False)

with tf.name_scope("SimNet"):
	net = mynet(img_placeholder, reuse=False)
	#variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	#print(variables)
	sim_saver = tf.train.Saver()
	sim_sess = tf.Session()
	sim_saver.restore(sim_sess, 'm3/m.ckpt')
#---------------------------------------------------------------------------------------
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

with tf.name_scope("SliNet"):
	model = NLDF.Model()
	model.build_model()
	sli_saver = tf.train.Saver()
	sli_sess = tf.Session()
	img_size = NLDF.img_size
	label_size = NLDF.label_size
	ckpt = tf.train.get_checkpoint_state('Model/')
	sli_saver.restore(sli_sess, ckpt.model_checkpoint_path)
	
img,img_shape = sliPreProcess(img_size)
result = sli_sess.run(model.Prob,feed_dict={model.input_holder: img})
sliPostProcess(result,label_size,img_shape)
	

#--------------------------------------------------------
img1 = simPreProcess(left_img)
img2 = simPreProcess(right_img)
left = sim_sess.run(net, feed_dict={img_placeholder:[img1]})
right = sim_sess.run(net, feed_dict={img_placeholder:[img2]})
dist = euclidean_distance(left,right)
d=dist.eval(session=sim_sess)
if d>=mean:
	d = 0.0
else:
	d = 1.0
	
print('output:',d)
