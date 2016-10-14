''' 
Author: Leong Wei Zhen
Description: Python interface to deploy the trained network (Sliding Approach) 
Further description: The trained model <.caffemodel> is deployed (either using only CPU or with GPU). The predicted coordinates of the eyes are plotted with the image.
			
The algorithm flow is described as follows:

1) apply train model to patches of the input image
2) compile the estimated results
3) applying thresholding to remove outliers then average (currently the threshold is set to 20) 
--> indicating any neighbouring pixel coordinate with 20 pixel distance apart will be clustered in a group
4) finalizing the output result
	       	
A new approach demonstrating immunity to spatial transformation (namely translation)		
'''

import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import sys
import os
import time
#import matplotlib.mlab as mlab
from scipy import stats

iteration = sys.argv[1]
file_name = sys.argv[2]
stride_no = int(sys.argv[3])

import numpy as np 
#import Image

# use grayscale output
#plt.rcParams['image.cmap'] = 'gray'

import sys
#caffe_root = '/home/wzleong/caffe'
#sys.path.insert(0, caffe_root + 'python')
import caffe

import os

#file_path= sys.argv[1]
if os.path.isfile('/home/wzleong/caffe/examples/BIOID_face/_iter_10000.caffemodel'):
	print 'BIOID found'
else: 
	print 'Error'

def get_coordinate(coordinates, x_max, y_max):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = (float(coordinates[i])*(x_max/2)) + (x_max/2)
		else: 
			coordinates[i] = (float(coordinates[i])*(y_max/2)) + (y_max/2)
	return coordinates


# the estimated results are only from patches of the image, so the coordinates must be shifted back with respect of the original image
def renormalize_coordinates(coordinates, shift_x, shift_y):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = float(coordinates[i]) + shift_x
		else: 
			coordinates[i] = float(coordinates[i]) + shift_y
	return coordinates

# function to get filename of input image 
# the filename will be used to save the result for better referencing
def get_filename(file_path):
	path, file_name = os.path.split(file_path)
	file_name = file_name.split('.', 1)[0]
	return file_name	

# function to group small cluster of the estimated coordinate 
def thresholding(data, threshold):
	return np.ceil(data/threshold)*threshold

# after applying the mask to remove outliers --> the finalize output will be the average of all the estimated coordinates
def estimate_average(data):
	count = 0 
	accumulate = 0 
	for i, value in enumerate(data):
		if value != 0:
			count += 1
			accumulate += value
			
	return accumulate/count


# set Caffe to run in GPU mode 
#caffe.set_device(0)
#caffe.set_mode_gpu()

# uncomment to set Caffe to run in CPU mode 
caffe.set_mode_cpu()

print "ok"

model_def_prototxt = '/home/wzleong/caffe/examples/BIOID_face/bioid_deploy_only_one.prototxt'
model_weights = '/home/wzleong/caffe/examples/BIOID_face/_iter_'+iteration+'.caffemodel'


#defines the CNN network 
start = time.time()
net = caffe.Net(model_def_prototxt, model_weights, caffe.TEST)		
#use test mode (not in train mode -> no dropout)
#model_weights, 	#containes the trained weights  

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# caffe reads the image into array of (28,28,1) 
# after transpose will convert the array to (1,28,28)
transformer.set_transpose('data', (2,0,1))
# normalize the valus of the image based on 0-255 range 
#transformer.set_raw_scale('data', 255)

net.blobs['data'].reshape(1,1,200,200)

#test_image = caffe.io.load_image(os.path.join("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_"+file_name+".pgm"), color=False)
test_image = caffe.io.load_image(os.path.join(file_name), color=False)

test_image = test_image.astype(float)
test_image = test_image - np.mean(test_image)

# cafe load image as H x W x N
height, width, number = test_image.shape
print test_image.shape

count_x = count_y = 0

img = mpimg.imread(file_name)

# first plot of input image
pl.subplot(311)
pl.imshow(img, cmap='gray' )
# second plot of images with estimated locations of the eye applying the 'sliding' approach
pl.subplot(312)
pl.imshow(img, cmap='gray' )


for i in range(stride_no):
	for j in range(stride_no):

		input_image = test_image[ count_y + j*((height - 200)/stride_no) : count_y +j*((height - 200)/stride_no) + 200, count_x + i*((width - 200)/stride_no) : count_x + i*((width - 200)/stride_no) + 200]	
		net.blobs['data'].data[0] = transformer.preprocess('data', input_image)
		output = net.forward()

		# the output probability for the first image (the tested image) 
		output_result = output['fc2'][0]

		output_result = get_coordinate(output_result,200,200)
		output_result = renormalize_coordinates(output_result, i*((width - 200)/stride_no), j*((height - 200)/stride_no))
		#print output_result
		# plotting all the estimated locations 
		# (might include wrong estimation because the trained model will slide through multiple locations of the image - including patches without eye)
		pl.scatter(output_result[0], output_result[1], marker='x', s=30)
		pl.scatter(output_result[2], output_result[3], marker='x', s=30)
		# print 'y:', count_y + j*((height - 200)/stride_no) 
		# print 'x:', count_x + j*((width - 200)/stride_no)
		
		if i == 0 and j == 0:
			output_result_compiled = output_result
		else: 
			output_result_compiled = np.concatenate((output_result_compiled,output_result), axis = 0)
			
end = time.time()
time_elapsed = (end - start) * 1000
print time_elapsed
print output_result_compiled.shape
output_result_compiled = output_result_compiled.reshape((i+1)*(j+1),4)
print output_result_compiled.shape
print output_result_compiled

# finalize the location of the eye from the compilation of eye coordinates
for i in range(4):
	temp = thresholding(output_result_compiled[:,i], 20) 
	dist_mode = stats.mode(temp)
	#print dist_mode[i]
	mask = temp 
	for j, value in enumerate(mask):
		if value != dist_mode[0]:
			mask[j] = 0
		else:
			mask[j] = 1

	#print mask
	#print output_result_compiled[:,i]

	#print np.multiply(output_result_compiled[:,i],mask)

	average = estimate_average(np.multiply(output_result_compiled[:,i],mask))
			
	if i == 0:
		average_compiled = np.array([average])
	else: 	
		average_compiled = np.append(average_compiled,average)

#print average_compiled

'''
###########################################################################
## Histogram plot of the distribution of the coordinates 
## ------------------------------------------------------------------------
## Uncomment when necessary 
## Provide visualization of the coordinate distribution 
###########################################################################

#pl.subplot(413)
#binwidth = 0.1
# round up to the the 2nd integer point (121 -> 130)
#n, bins, patches = pl.hist(np.around(output_result_compiled[:,0], decimals = -1), bins=np.arange(min(np.around(output_result_compiled[:,0], decimals = -1)), max(np.around(output_result_compiled[:,0], decimals = -1)) + binwidth, binwidth), normed=1)
#pl.title('Histogram plot of inferencing time \n for each image at ' + iteration + 'iteration')
'''

pl.subplot(313)
pl.imshow(img, cmap='gray' )			
pl.scatter(average_compiled[0], average_compiled[1], marker='x', s=30)
pl.scatter(average_compiled[2], average_compiled[3], marker='x', s=30)



#pl.show()
file_name = get_filename(file_name)
pl.savefig('testing'+file_name+'.jpg',dpi=500)






