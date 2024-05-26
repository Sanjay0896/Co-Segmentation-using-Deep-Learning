import os
import tensorflow as tf

model_dir = '/home/sanjay/Desktop/Project/Model'
label = 'sli'
dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir):
	if not tf.gfile.Exists(model_dir):
		raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)
			
	checkpoint = tf.train.get_checkpoint_state(model_dir)
	input_checkpoint = checkpoint.model_checkpoint_path
	
	absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
	output_graph = absolute_model_dir + '/frozen_model_'+label+'.pb'
	print('freezed model stored at ',output_graph)
	
	
	# We clear devices to allow TensorFlow to control on which device it will load operations
	clear_devices = True
	
	with tf.Session(graph=tf.Graph()) as sess:
		saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
		graph = tf.get_default_graph()
		input_graph_def = graph.as_graph_def()
		saver.restore(sess, input_checkpoint)
		
		'''file = open('varlist.txt','w')
		for n in tf.get_default_graph().as_graph_def().node:
			print(n.name,' ',n.op)
			file.write(n.name+'          '+n.op+' \n')
		#sys.exit(0)
		file.close()
		os._exit(0)'''
		
		#output_node_names =  'model/Flatten/Reshape'
		#output_node_names =  'middle,model/Flatten/Reshape'
		output_node_names =  'Softmax'
		
		output_graph_def = tf.graph_util.convert_variables_to_constants(
				sess,input_graph_def, 
				output_node_names.split(",") # The output node names are used to select the usefull nodes
			)
			
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(output_graph_def.SerializeToString())
		print("%d ops in the final graph." % len(output_graph_def.node))
	
	return output_graph_def
	
freeze_graph(model_dir)
