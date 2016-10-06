'''
Author: Leong Wei Zhen
Description: Python code to shift the image sideways
'''

import cv2
import numpy as np
import sys 

number = sys.argv[1]

img = cv2.imread('BioID-FaceDatabase-V1.2/BioID_'+number+'.pgm',0)
rows,cols = img.shape

#print rows
#print cols
 
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('BioID_'+number+'_resize.pgm',dst)
