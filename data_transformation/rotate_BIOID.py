''' 
Author: Leong Wei Zhen 
Description: Python code to rotate the BIOID image 
Dependencies: 	1) OpenCV
'''

import cv2
import numpy as np
import sys 

number = sys.argv[1]

img = cv2.imread('BioID-FaceDatabase-V1.2/BioID_'+number+'.pgm',0)
rows,cols = img.shape

#print rows
#print cols

# rotate the image 90 degress anti-clockwise
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))


cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('BioID_'+number+'_rotate.pgm',dst)
