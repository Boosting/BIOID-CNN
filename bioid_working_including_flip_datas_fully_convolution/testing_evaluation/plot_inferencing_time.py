import numpy as np 
import matplotlib.pyplot as plt
import caffe
import matplotlib.image as mpimg
import sys
import math
import os
import time

iteration = sys.argv[1]

def get_coordinate(coordinates, x_max, y_max):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = (float(coordinates[i])*(x_max/2)) + (x_max/2)
		else: 
			coordinates[i] = (float(coordinates[i])*(y_max/2)) + (y_max/2)
	return coordinates


def run_test(file_name):
	
	# setting up caffe
	caffe.set_mode_cpu()

	model_def_prototxt = '/home/wzleong/caffe/examples/BIOID_face/bioid_working_including_flip_datas_fully_convolution/bioid_deploy_only_one.prototxt'
	model_weights = '/home/wzleong/caffe/examples/BIOID_face/bioid_working_including_flip_datas_fully_convolution/_iter_'+iteration+'.caffemodel'

	start = time.time()
	#defines the CNN network 
	net = caffe.Net(model_def_prototxt, model_weights, caffe.TEST)		
	#use test mode (not in train mode -> no dropout)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	# caffe reads the image into array of (28,28,1) 
	# after transpose will convert the array to (1,28,28)
	transformer.set_transpose('data', (2,0,1))
	# normalize the valus of the image based on 0-255 range 
	#transformer.set_raw_scale('data', 255)

	net.blobs['data'].reshape(1,1,286,384)

	test_image = caffe.io.load_image(os.path.join("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_"+file_name+".pgm"), color=False)
	net.blobs['data'].data[0] = transformer.preprocess('data', test_image)
	output = net.forward()

	output_result = output['conv7'][0]

	output_result = get_coordinate(output_result,384,286)
	end = time.time()
	time_elapsed = (end - start) * 1000
	time_elapsed = time_elapsed

	return time_elapsed

 
def plot_error():
	
	time_elapsed = list()
	binwidth = 0.001

	for i in range(0,1521):
		print i		
		number = str(i).zfill(4)
		time_elapsed_temp = run_test(number)
		time_elapsed.append(time_elapsed_temp)
	

	# If 'normed' = True, the first element of the return tuple will be the counts normalized to form a probability density, i.e., n/(len(x)`dbin), 
	# i.e., the integral of the histogram will sum to 1. If stacked is also True, the sum of the histograms is normalized to 1.
	plt.hist(time_elapsed, bins=np.arange(min(time_elapsed), max(time_elapsed) + binwidth, binwidth), normed=1)
	plt.title('Histogram plot of inferencing time \n for each image at ' + iteration + 'iteration')
	plt.xlabel('Inferencing time, ms')
	plt.ylabel('Normalized probability')
	


	plt.savefig('histogram_inferencing_time_iteration_'+iteration+'.jpg',bbox_inches='tight')



plot_error()


	




