
# command to run: 
# python python_eye_hdf5.py data/BIO_ID_face/python_eye_hdf5.py data/BIO_ID_face/BioID-FaceDatabase-V1.2/ data/BIO_ID_face/BioID-FaceDatabase-V1.2/
# python python_eye_hdf5.py BioID-FaceDatabase-V1.2/ BioID-FaceDatabase-V1.2/

import os
import sys
#import caffe
import cv2
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
x_max = 286
y_max = 384
# channel = 1 for grayscale, channel = 3 for RGB
channel = 1 
FTEST = 'hello'
FTRAIN = 'hello1'


def get_coordinate(coordinates, x_max, y_max):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = (float(coordinates[i])*(x_max/2)) + (x_max/2)
		else: 
			coordinates[i] = (float(coordinates[i])*(y_max/2)) + (y_max/2)
	return coordinates



def load_single_img(img_path, eye_path):
	with open(eye_path,'r') as f: 
		for i, line in enumerate(f):
			#print line
			if i == 1:
				eye_coordinates = line.split()

				# flip coordinates
				eye_coordinates_flipped = eye_coordinates
				eye_coordinates_flipped[2] = 384/2 - (float(eye_coordinates[2]) - 384/2)
				eye_coordinates_flipped[0] = 384/2 + (384/2 - float(eye_coordinates[0]))

				# scale the eye_coordinates to fit [-1,1]
				eye_coordinates_flipped = scale_coordinates(eye_coordinates_flipped, x_max, y_max)
				#print eye_coordinates 
				#print eye_coordinates


	# Code to display input image and print results
	#image = caffe.io.load_image(image_dir, color=False)
	#img=mpimg.imread(img_path)	
	#print img

	# flip image
	img=cv2.imread(img_path,0)
	#print img
	img=cv2.flip(img,1)
	#print img.shape	

	img = img.astype(np.float32)	
	img=scale_pixels(img,255)
	#print img
	image_dim = img.shape
	#print image_dim
	
	#imgplot = plt.imshow(img, cmap='gray')
	#plt.annotate('here',xy=(232,110),xycoords='figure fraction') 
	#plt.show()

	return img, eye_coordinates_flipped

def scale_coordinates(coordinates, x_max, y_max):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			#coordinates[i] = (float(coordinates[i]) - (286/2))/(286/2)
			coordinates[i] = (float(coordinates[i]) - (x_max/2))/(x_max/2)
		else: 
			#coordinates[i] = (float(coordinates[i]) - (384/2))/(384/2)
			coordinates[i] = (float(coordinates[i]) - (y_max/2))/(y_max/2)
	return coordinates

def scale_pixels(image, max_pixel):
	image = image/255
	return image

def load(image_dir,eye_dir,test=False): 
	fname = FTEST if test else FTRAIN 
	for i in range(0,1521):
		print i		
		number = str(i).zfill(4)
		#image_path = os.path.join(image_dir,"BioID_" + number + ".pgm")
		#print image_path
		image, eye_coordinate = load_single_img(os.path.join(image_dir,"BioID_" + number + ".pgm"), os.path.join(image_dir,"BioID_" + number + ".eye"))

		#print image.shape
		#if i == 0:		
			#print eye_coordinate
			#print image
		if i == 0:
			image_compiled = image
			eye_coordinate_compiled = eye_coordinate
		else: 
			#image_compiled = np.array([image,image_compiled])
			#eye_coordinate_compiled = np.array([eye_coordinate, eye_coordinate_compiled])
			image_compiled = np.concatenate((image_compiled,image), axis = 0)
			eye_coordinate_compiled = np.concatenate((eye_coordinate_compiled,eye_coordinate), axis = 0)
		#print image_compiled.shape
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


		

'''
###############################	
#Test
#####################################
image,eye_coordinates = load_single_img(os.path.join(image_dir,"BioID_0010.pgm"), os.path.join(image_dir,"BioID_0010.eye"))
#print eye_coordinates
eye_coordinates = get_coordinate(eye_coordinates,286,384)
#print image
plt.imshow(image, cmap='gray' )
plt.scatter(eye_coordinates[0], eye_coordinates[1], marker='x', s=30)
plt.scatter(eye_coordinates[2], eye_coordinates[3], marker='x', s=30)
plt.show()
'''

#image,eye_coordinates = load_single_img(os.path.join(image_dir,"BioID_0001.pgm"), os.path.join(image_dir,"BioID_0001.eye"))
#print eye_coordinates
#print image

#print eye_coordinates
#print image
#image_dim = image.shape
#print image_dim

# total number of images for training or validation respectively
# 
#image = image.reshape((channel,286,384))
#image_dim = image.shape
#print image_dim


image_compiled, eye_coordinate_compiled = load(image_dir, eye_dir)

'''
########################
# testing
########################
eye_coordinate_compiled[2] = get_coordinate(eye_coordinate_compiled[2],286,384)
#print image
plt.imshow(image_compiled[2][0], cmap='gray' )
plt.scatter(eye_coordinate_compiled[2][0], eye_coordinate_compiled[2][1], marker='x', s=30)
plt.scatter(eye_coordinate_compiled[2][2], eye_coordinate_compiled[2][3], marker='x', s=30)
plt.show()
'''


image_compiled, eye_coordinate_compiled = shuffle(image_compiled, eye_coordinate_compiled, random_state=42)
#print image_compiled[0]
#print image_compiled[0].shape
sep = 1300
#print image_compiled[0:sep].shape
#print image_compiled[sep:].shape
writeHdf5('train_scaled_flip',image_compiled[0:sep],eye_coordinate_compiled[0:sep])
writeHdf5('val_scaled_flip',image_compiled[sep:1500],eye_coordinate_compiled[sep:1500])













