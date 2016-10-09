'''
Author: Leong Wei Zhen
Description: 	Python code to shift the image sideways
		Utilizing OpenCV affine transformation with addition argument "borderMode=cv2.BORDER_REPLICATE" for extrapolation at the border 
		(to remove discontinuity or smoothen the edges)

Command to run in terminal: 	python translation_BIOID.py <BioID number> <shift_x_displacement> <shift_y_displacement>
				example: python translation_BIOID.py 0000 -100 50

'''

import cv2
import numpy as np
import sys 
import matplotlib.pyplot as plt 

number = sys.argv[1]
shift_x = sys.argv[2]
shift_y = sys.argv[3]

img = cv2.imread('BioID-FaceDatabase-V1.2/BioID_'+number+'.pgm',0)
rows,cols = img.shape

#print rows
#print cols

def translate_image_and_annotation (image_path, annotation_path, shift_x, shift_y):
	# openCV code to shift the image by (shift_x, shift_y) ==> mapping matrix: [1,0,shift_x],[0,1,shift_y]
	# with reference to: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
	img = cv2.imread(image_path,0)
	M = np.float32([[1,0,shift_x],[0,1,shift_y]])
	image=cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
	
	#########################################
	# read annotations
	#########################################
	with open(annotation_path,'r') as f: 
		for i, line in enumerate(f):
			if i == 1:
				eye_coordinates = line.split()
		
	for i, value in enumerate(eye_coordinates):
		if i%2 == 0: 
			eye_coordinates[i] = float(eye_coordinates[i]) + shift_x
		else: 
			eye_coordinates[i] = float(eye_coordinates[i]) + shift_y

	return image, eye_coordinates

image, eye_coordinates = translate_image_and_annotation ('BioID-FaceDatabase-V1.2/BioID_'+number+'.pgm', 'BioID-FaceDatabase-V1.2/BioID_'+number+'.eye', int(shift_x), int(shift_y))
plt.imshow(image, cmap='gray' )
plt.scatter(eye_coordinates[0], eye_coordinates[1], marker='x', s=30)
plt.scatter(eye_coordinates[2], eye_coordinates[3], marker='x', s=30)
plt.show()
	



'''
################################################################################################
### Original code tested for translation
################################################################################################
# shift(100,50)
M = np.float32([[1,0,100],[0,1,50]])
dst=cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('BioID_'+number+'_resize_bottom_right.pgm',dst)

# shift(-100,50)
M = np.float32([[1,0,-100],[0,1,50]])
dst=cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('BioID_'+number+'_resize_bottom_left.pgm',dst)	

# shift(100,50)
M = np.float32([[1,0,100],[0,1,-50]])
dst=cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('BioID_'+number+'_resize_top_right.pgm',dst)

# shift(100,50)
M = np.float32([[1,0,-100],[0,1,-50]])
dst=cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('BioID_'+number+'_resize_top_left.pgm',dst)

'''



