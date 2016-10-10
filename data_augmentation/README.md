# Demonstratation of data augmentation (preprocessing)

The images are produced from the Python codes on BIOID-CNN/data_transformation/ : 
1. Flip horizontally 
2. Rotation 
3. Translation (shifting the image by pixel of (x, y))

Note: OpenCV offers a flag to extrapolate the border pixels during affine transformation: borderMode=cv2.BORDER_REPLICATE
This is enabled to avoid huge discontinuity at the borders which often causes the occurance many false positive 
at the initial convolution layer which are sensitive to high level features such as edges. 

Further reference on OpenCV:
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html


