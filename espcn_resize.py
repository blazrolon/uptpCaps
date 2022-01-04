# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:20:54 2021

@author: biabe
"""
import cv2
import get_latest
import matplotlib.pyplot as plt

img=cv2.imread(get_latest.get_last_image(),1)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "ESPCN_x3.pb"
sr.readModel(path)
sr.setModel("espcn",3)
result = sr.upsample(img)
# Resized image
resized = cv2.resize(img,dsize=None,fx=3,fy=3)
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)
# Original image
plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2)
# SR upscaled
plt.imshow(result[:,:,::-1])
plt.subplot(1,3,3)
# OpenCV upscaled
plt.imshow(resized[:,:,::-1])
plt.show()
cv2.imwrite('result.jpeg',resized)
