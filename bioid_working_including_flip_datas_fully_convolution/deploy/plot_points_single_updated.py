''' 
Author: Leong Wei Zhen
Description: Python interface to deploy the trained network 
Further description: The trained model <.caffemodel> is deployed (either using only CPU or with GPU). The predicted coordinates of the eyes are plotted with the image.
			
*************** Only difference from the previous model is that the output layer is from a convolution layer (conv7) since it is a fully convolution network
'''

import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import sys
import os

iteration = sys.argv[1]
file_name = sys.argv[2]
read_file = False

import numpy as np 

import sys
import caffe

import os

#file_path= sys.argv[1]
if os.path.isfile('/home/wzleong/caffe/examples/BIOID_face/bioid_working_including_flip_datas_fully_convolution/_iter_10000.caffemodel'):
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

# set Caffe to run in GPU mode 
#caffe.set_device(0)
#caffe.set_mode_gpu()

# uncomment to set Caffe to run in CPU mode 
caffe.set_mode_cpu()

print "ok"

model_def_prototxt = '/home/wzleong/caffe/examples/BIOID_face/bioid_working_including_flip_datas_fully_convolution/bioid_deploy_only_one.prototxt'
model_weights = '/home/wzleong/caffe/examples/BIOID_face/bioid_working_including_flip_datas_fully_convolution/_iter_'+iteration+'.caffemodel'


#defines the CNN network 
net = caffe.Net(model_def_prototxt, model_weights, caffe.TEST)		
#use test mode (not in train mode -> no dropout)
#model_weights, 	#containes the trained weights  

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
print output_result

img = mpimg.imread(os.path.join("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_"+file_name+".pgm"))
#img = mpimg.imread(os.path.join("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_000.pgm"))
pl.imshow(img, cmap='gray' )
#print coordinates
pl.scatter(output_result[0], output_result[1], marker='x', s=30)
pl.scatter(output_result[2], output_result[3], marker='x', s=30)
# code to display the image with the predicted coordinate of eye
pl.show()
# code to save the image with the predicted coordinate of eye into jpeg file
# pl.savefig('result_image_'+iteration+'_'+'bioid'+file_name+'.jpg')

		

