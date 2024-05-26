import tensorflow as tf
import cv2
import sys
import numpy as np
import os

frozen_model_sli_filename = 'frozen_model_sli.pb'
save_dir = 'result'
VGG_MEAN = [103.939, 116.779, 123.68]
image_size = 480

frozen_model_sim_filename = 'frozen_model_sim.pb'
mean = 1.3523

left_img = sys.argv[1]
right_img = sys.argv[2]

lst = left_img.split('/')
left_image_name = lst[len(lst)-1].split('.')[0]
left_ext = lst[len(lst)-1].split('.')[1]

lst = right_img.split('/')
right_image_name = lst[len(lst)-1].split('.')[0]
right_ext = lst[len(lst)-1].split('.')[1]

#----------------------------------Defination-------------------------------------------
def load_graph(frozen_graph_filename):
    
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,input_map=None,return_elements=None, name="")
    return graph

def sliPreProcess(img_path,img_size):
	img = cv2.imread(img_path)
	img_shape = img.shape
	img = cv2.resize(img, (img_size, img_size)) - VGG_MEAN
	img = img.reshape((1, img_size, img_size, 3))
	return img,img_shape

def sliPostProcess(img_name,result,label_size,img_shape):
	result = np.reshape(result, (label_size, label_size, 2))
	result = result[:, :, 0]
	result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))
	#f = img_path
	#fn = f.split('.')
	save_name = os.path.join(save_dir,img_name+'_Salient.png')       
	cv2.imwrite(save_name, (result*255).astype(np.uint8))
	
def simPreProcess(img_path):
	img = cv2.imread(img_path)
	img = np.divide(img,255.0)
	img = cv2.resize(img,(image_size,image_size))
	return img

def euclidean_distance(x, y):
    d = tf.square(tf.subtract(x, y))
    return tf.sqrt(tf.reduce_sum(d,1))

def cropping(img,img2):
    #print(img2)
    mask = img > 0
    coords = np.argwhere(mask)
    p1 = coords.min(axis=0)
    p2 = coords.max(axis=0)+1
    #print(p1,p2)
    crop = img2[p1[0]:p2[0],p1[1]:p2[1]]
    crop = cv2.resize(crop,(image_size,image_size))
    return crop
#--------------------------------Saliency----------------------------------------------
sli_graph = load_graph(frozen_model_sli_filename)

sli_img_placeholder = sli_graph.get_tensor_by_name('Placeholder:0')
SliNet = sli_graph.get_tensor_by_name('Softmax:0')

img_size = 352
label_size = img_size / 2
with tf.Session(graph=sli_graph) as sess:
	img,img_shape = sliPreProcess(left_img,img_size)
	result = sess.run(SliNet,feed_dict={sli_img_placeholder: img})
	sliPostProcess(left_image_name,result,label_size,img_shape)
	
	img,img_shape = sliPreProcess(right_img,img_size)
	result = sess.run(SliNet,feed_dict={sli_img_placeholder: img})
	sliPostProcess(right_image_name,result,label_size,img_shape)
	
#--------------------------------middle----------------------------------------------
sli_img = save_dir+'/'+left_image_name+'_Salient.png'
img1 = cv2.imread(sli_img)
img2 = cv2.imread(left_img)
crp_img = cropping(img1,img2)
left_croped = save_dir+'/'+left_image_name+'_crop.'+left_ext
cv2.imwrite(left_croped,crp_img)

sli_img = save_dir+'/'+right_image_name+'_Salient.png'
img1 = cv2.imread(sli_img)
img2 = cv2.imread(right_img)
crp_img = cropping(img1,img2)
right_croped = save_dir+'/'+right_image_name+'_crop.'+right_ext
cv2.imwrite(right_croped,crp_img)

#--------------------------------Siamese----------------------------------------------
sim_graph = load_graph(frozen_model_sim_filename)


sim_img_placeholder = sim_graph.get_tensor_by_name('middle:0')
SimNet = sim_graph.get_tensor_by_name('model/Flatten/Reshape:0')

with tf.Session(graph=sim_graph) as sess:
	img1 = simPreProcess(left_croped)
	left = sess.run(SimNet, feed_dict={sim_img_placeholder:[img1]})
	
	img2 = simPreProcess(right_croped)
	right = sess.run(SimNet, feed_dict={sim_img_placeholder:[img2]})
	
	dist = euclidean_distance(left,right)
	d=dist.eval()
	print(d)
	if d>=mean:
		d = 0.0
	else:
		d = 1.0
	
	print('output:',d)
