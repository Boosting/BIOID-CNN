import numpy as np 
import matplotlib.pyplot as plt
import caffe
import matplotlib.image as mpimg
import sys
import math
import os

iteration = sys.argv[1]
label_type = sys.argv[2]
threshold = sys.argv[3]

def calculate_error(ground_truth, output_result, label_type):
	'''
	####################################
	# demo 
	####################################
	a = np.array([1,2,2,2])
	print a 
	b = np.array([5,3,3,4])
	print b
	# left eye
	diff = np.linalg.norm(b[0:2]-a[0:2])
	print diff
	diff = math.sqrt( pow( (float(a[0]) - float(b[0])) ,2) + pow( (float(a[1]) - float(b[1])) ,2) )
	# right eye
	diff = np.linalg.norm(b[2:4]-a[2:4])
	print diff
	'''
	if label_type is 'left': 
		error = math.sqrt( pow( (float(output_result[0]) - float(ground_truth[0])) ,2) + pow( (float(output_result[1]) - float(ground_truth[1])) ,2) )
	else: 
		error = math.sqrt( pow( (float(output_result[2]) - float(ground_truth[2])) ,2) + pow( (float(output_result[3]) - float(ground_truth[3])) ,2) )
	
   	print 'error', error
	return error



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

	model_def_prototxt = '/home/wzleong/caffe/examples/BIOID_face/bioid_working_including_flip_datas/bioid_deploy_only_one.prototxt'
	model_weights = '/home/wzleong/caffe/examples/BIOID_face/bioid_working_including_flip_datas/_iter_'+iteration+'.caffemodel'


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

	output_result = output['fc2'][0]

	output_result = get_coordinate(output_result,384,286)
	#print output_result

	return output_result

def get_ground_truth(file_name):
	with open("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_"+file_name+".eye",'r') as f: 
		for i, line in enumerate(f):
			if i == 1:
				eye_coordinates = line.split()
	return eye_coordinates
 
def plot_error(label_type):
	
	error = list()
	
	# Setting for the plot
	#plt.rc('font', size=14)	
	#fig = plt.figure(figsize=(20, 15))
	binwidth = 0.001
	#yCut = np.linspace(0, 70, 100)
	#xCut = np.ones(100)*0.05	

	for i in range(0,1521):
		print i		
		number = str(i).zfill(4)
		output_result = run_test(number)
		ground_truth = get_ground_truth(number)
		error.append(calculate_error(ground_truth, output_result, label_type))
	
	'''
	code to demonstrate usage of sort
	a = [1,2,6,3,4,8]
	c = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:<number of elements>]
	print c
	Result > [5, 2, 4] 
	'''
	
	# obtain the number of outlier > $threshold
	outlier_number = sum(i > int(threshold) for i in error)
	# get the BioID number of those with euclidean distance error greater than $threshold
	outliers = sorted(range(len(error)), key=lambda i: error[i], reverse=True)[:outlier_number]

	# set position of plot to make a bit of room for extra text
	# gca().set_position( x0, y0, width, height)
	plt.gca().set_position((.1, .3, .8, .7)) 


	# If 'normed' = True, the first element of the return tuple will be the counts normalized to form a probability density, i.e., n/(len(x)`dbin), 
	# i.e., the integral of the histogram will sum to 1. If stacked is also True, the sum of the histograms is normalized to 1.
	plt.hist(error, bins=np.arange(min(error), max(error) + binwidth, binwidth), normed=1)
	#ax.plot(xCut, yCut, 'r', linewidth=2)
	#fig.set_title('left eye')
	plt.title('Histogram plot of euclidean distance error distribution \nof ' + label_type + ' eye coordinates at ' + iteration + 'iteration')
	plt.xlabel('Euclidean distance error')
	plt.ylabel('Normalized probability')
	
	outlier_string = ""
	for index, i in enumerate(outliers):
		if index%7 == 0:
			outlier_string = outlier_string + '\n' + 'BioID' + str(outliers[index]).zfill(4) + ' '
		else: 
			outlier_string = outlier_string + 'BioID' + str(outliers[index]).zfill(4) + ' '
		 
	plt.figtext(.02, .02, "The outliers which has Euclidean distance error (E >" +threshold+ ") are: " + outlier_string )

	plt.savefig('error_distribution_'+label_type+'_with_threshold_'+threshold+'_iteration_'+iteration+'.jpg',bbox_inches='tight')

	
	
	return outlier_number, outliers


outlier_number, outliers = plot_error(label_type)
print outlier_number
print outliers


	




