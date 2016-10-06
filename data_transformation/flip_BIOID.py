''' 
Author: Leong Wei Zhen 
Description: Python code to flip the BIOID image horizontally 
Dependencies: 	1) OpenCV
		2) Matplotlib Python package
'''

# command to be run in terminal 
# python flip_BIOID.py <BIOID_number>

import cv2
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import sys 
import os

number = sys.argv[1]

# read the image file
img = cv2.imread('BioID-FaceDatabase-V1.2/BioID_'+number+'.pgm',0)

# will flip the image horizontally 
flip_img=cv2.flip(img,1)

# write a copy of the flipped image
cv2.imwrite('BioID_'+number+'_flip.pgm',flip_img)

# open the anootation (coordinates of the eye)
with open(os.path.join("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_"+number+".eye"),'r') as f: 
	for i, line in enumerate(f):
		if i == 1:
			eye_coordinates = line.split()

pl.imshow(flip_img, cmap='gray' )

# update the coordinates of the eye (after flipping)
eye_coordinates_flipped = eye_coordinates
eye_coordinates_flipped[2] = 384/2 - (float(eye_coordinates[2]) - 384/2)
eye_coordinates_flipped[0] = 384/2 + (384/2 - float(eye_coordinates[0]))

# plot and display the coordinates of the updated eye
pl.scatter(eye_coordinates_flipped[0], eye_coordinates_flipped[1], marker='x', s=30)
pl.scatter(eye_coordinates_flipped[2], eye_coordinates_flipped[3], marker='x', s=30)
pl.show()
