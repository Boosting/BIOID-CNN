import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import sys
import os
import time

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



def renormalize_coordinates(coordinates, shift_x, shift_y):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = float(coordinates[i]) + shift_x
		else: 
			coordinates[i] = float(coordinates[i]) + shift_y
	return coordinates

def get_filename(file_path):
	path, file_name = os.path.split(file_path)
	file_name = file_name.split('.', 1)[0]
	return file_name	


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

#img = mpimg.imread(os.path.join("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_"+file_name+".pgm"))
img = mpimg.imread(file_name)

#pl.figure(figsize=(780,600))
pl.subplot(211)
pl.imshow(img, cmap='gray' )
pl.subplot(212)

#img = mpimg.imread(os.path.join("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_000.pgm"))
pl.imshow(img, cmap='gray' )
#print coordinates

for i in range(stride_no):
	for j in range(stride_no):

		input_image = test_image[ count_y + j*((height - 200)/stride_no) : count_y +j*((height - 200)/stride_no) + 200, count_x + i*((width - 200)/stride_no) : count_x + i*((width - 200)/stride_no) + 200]	
		net.blobs['data'].data[0] = transformer.preprocess('data', input_image)
		output = net.forward()

		# the output probability for the first image (the tested image) 
		output_result = output['fc2'][0]

		output_result = get_coordinate(output_result,200,200)
		output_result = renormalize_coordinates(output_result, i*((width - 200)/stride_no), j*((height - 200)/stride_no))
		print output_result
		pl.scatter(output_result[0], output_result[1], marker='x', s=30)
		pl.scatter(output_result[2], output_result[3], marker='x', s=30)
		print 'y:', count_y + j*((height - 200)/stride_no) 
		print 'x:', count_x + j*((width - 200)/stride_no)
		
		if i == 0 and j == 0:
			output_result_compiled = output_result
		else: 
			output_result_compiled = np.concatenate((output_result_compiled,output_result), axis = 0)
		
		
end = time.time()
time_elapsed = (end - start) * 1000
print time_elapsed

#pl.show()
file_name = get_filename(file_name)
pl.savefig('plot_'+file_name+'.jpg',dpi=500)






