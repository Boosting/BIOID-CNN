'''
Author: Leong Wei Zhen
Description: Python code to read images and parse the eye coordinates then store them into HDF5 database

Futher Description:	Images - read using matplotlib with pixel values range [0,255] --> rescale to [0,1]
			Eye coordinates - read from .eye files and stored as list (1d matrix) with values range [384,286] --> rescale to [-1,1]

Update from previous errors: previously the image was mistaken to be 286 x 384 (because matplotlib reads in graysale images and store them as 2D array in H x W format rather than W x H)

*** Further note: Caffe input images as N x C x H x W (which matches the input format from matplotlib)

'''

# command to run: 
# python python_eye_hdf5.py BioID-FaceDatabase-V1.2/ BioID-FaceDatabase-V1.2/

import os
import sys
#import caffe
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import h5py

'''
format stored for the eye coordinates
#LX     LY      RX      RY
232     110     161     110

Code to read from file and parse the eye coordinates
'''

image_dir = sys.argv[1]
eye_dir = sys.argv[2]
x_max = 384
y_max = 286
# channel = 1 for grayscale, channel = 3 for RGB
channel = 1 

# subroutine to load single image with the annotation 
def load_single_img(img_path, eye_path):
	with open(eye_path,'r') as f: 
		for i, line in enumerate(f):
			#print line
			if i == 1:
				eye_coordinates = line.split()
				# scale the eye_coordinates to fit [-1,1]
				eye_coordinates = scale_coordinates(eye_coordinates, x_max, y_max)

	# Code to display input image and print results
	img=mpimg.imread(img_path)	
	img = img.astype(np.float32)	
	img=scale_pixels(img,255)
	
	return img, eye_coordinates

# scale the coordinates of the eye to [-1,1]
def scale_coordinates(coordinates, x_max, y_max):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = (float(coordinates[i]) - (x_max/2))/(x_max/2)
		else: 
			coordinates[i] = (float(coordinates[i]) - (y_max/2))/(y_max/2)
	return coordinates

# scale the image pixel values to [0,1]
def scale_pixels(image, max_pixel):
	image = image/255
	return image

# load all the images provided by BIOID data set 
def load(image_dir,eye_dir,test=False): 
	fname = FTEST if test else FTRAIN 
	for i in range(0,1521):
		print i		
		number = str(i).zfill(4)
		image, eye_coordinate = load_single_img(os.path.join(image_dir,"BioID_" + number + ".pgm"), os.path.join(image_dir,"BioID_" + number + ".eye"))

		if i == 0:
			image_compiled = image
			eye_coordinate_compiled = eye_coordinate
		else: 
			image_compiled = np.concatenate((image_compiled,image), axis = 0)
			eye_coordinate_compiled = np.concatenate((eye_coordinate_compiled,eye_coordinate), axis = 0)
	image_compiled = image_compiled.reshape(1521,1,286,384)
	eye_coordinate_compiled = eye_coordinate_compiled.reshape(1521,4)
	return image_compiled, eye_coordinate_compiled
		

def writeHdf5(t,data,label=None):
	with h5py.File(os.getcwd()+ '/'+t + '_data.h5', 'w') as f:
		f['data'] = data
		if label is not None:
			f['label'] = label
	with open(os.getcwd()+ '/'+t + '_data_list.txt', 'w') as f:
		f.write(os.getcwd()+ '/' +t + '_data.h5\n')

def get_coordinate(coordinates, x_max, y_max):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = (float(coordinates[i])*(x_max/2)) + (x_max/2)
		else: 
			coordinates[i] = (float(coordinates[i])*(y_max/2)) + (y_max/2)
	return coordinates
		

'''
############### For verification purposes ###############
image,eye_coordinates = load_single_img(os.path.join(image_dir,"BioID_0019.pgm"), os.path.join(image_dir,"BioID_0019.eye"))
eye_coordinates = get_coordinate(eye_coordinates,384,286)

plt.imshow(image, cmap='gray' )
plt.scatter(eye_coordinates[0], eye_coordinates[1], marker='x', s=30)
plt.scatter(eye_coordinates[2], eye_coordinates[3], marker='x', s=30)
plt.show()
'''

image_compiled, eye_coordinate_compiled = load(image_dir, eye_dir)
# shuffle the image and eye coordinates to produce 2 sets for training and validation 
image_compiled, eye_coordinate_compiled = shuffle(image_compiled, eye_coordinate_compiled, random_state=42)
sep = 1300
writeHdf5('train_scaled',image_compiled[0:sep],eye_coordinate_compiled[0:sep])
writeHdf5('val_scaled',image_compiled[sep:1500],eye_coordinate_compiled[sep:1500])













