# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 13:12:16 2021

@author: biancarolon
"""
import imutils
import cv2
import get_latest

#import image file
img = cv2.imread(get_latest.get_last_image(),1)
#cv2.imshow('Original', img)
#cv2.destroyAllWindows()

#rescaling so image won't look distorted
r = 150.0 / img.shape[1]
dim = (150, int(img.shape[0] * r))
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

r = 50.0 / img.shape[0]
dim = (int(img.shape[1] * r), 50)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.waitKey(0)

#use imutils to maintain aspect ratio
resized = imutils.resize(img, width=100)
cv2.waitKey(0)
resized = imutils.resize(img, height=75)

methods = [('Upscaled', cv2.INTER_CUBIC)]

for (name, method) in methods:
	# increase the size of the image by 3x using the current
	# interpolation method
	print('[INFO] {}'.format(name))
	resized = imutils.resize(img, width=img.shape[1] * 2,inter=method) #upsampling
    #imS = cv2.resize(resized, (960, 540)) 
cv2.imwrite('output.jpeg', resized)
	#cv2.imshow('{}'.format(name), resized)
    
	#cv2.waitKey(0) #resizing result display

#cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
#cv2.imshow('finalImg',img)

cv2.destroyAllWindows()
