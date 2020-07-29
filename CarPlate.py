# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:13:35 2020

@author: himad


A Blob is a group of connected pixels in an image that share some common 
property ( E.g grayscale value ). 
In the image above, 
the dark connected regions are blobs, and the goal of blob detection 
is to identify and mark these regions.
"""

import cv2
import numpy as np
from transform import four_point_transform
from skimage.filters import threshold_local
from collections import namedtuple
from skimage import measure
from skimage import segmentation

LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])


image = cv2.imread("car.jpg")
ori = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
img = cv2.Canny(gray, 180, 200)

cnts, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:30]
number= None
cv2.imshow("ed", img)
count=0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx)==4:
        number = approx
        break
    
cv2.drawContours(image, [number], -1, (0,0, 255), 3)
warped= four_point_transform(ori, number.reshape(4,2))
V = cv2.split(cv2.cvtColor(warped, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 29, offset=15, method="gaussian")
thresh = (V>T).astype("uint8") * 255
thresh = cv2.bitwise_not(thresh)


labels = measure.label(thresh, connectivity=2, background=0)
charCandidates = np.zeros(thresh.shape, dtype="uint8")
print("Number of components:", np.max(labels))
print(labels.shape)

for label in np.unique(labels):
    if label==0:
        continue
    
    labelmask = np.zeros(thresh.shape, dtype="uint8")
    labelmask[labels==label] =255
    cnts,hierarchy = cv2.findContours(labelmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts)>0:
        
        c= max(cnts, key = cv2.contourArea)
        (bx,by, bw, bh) = cv2.boundingRect(c)
        
        aspectRatio = bw / float(bh)
        
        solidity = cv2.contourArea(c)/float(bw * bh)
        
        heightRatio = bh/ float(thresh.shape[0])

        keepAspectRatio = aspectRatio < 1.0
        keepSolidity = solidity > 0.15
        keepHeight = heightRatio > 0.4 and heightRatio < 0.95
 
				# check to see if the component passes all the tests
		
        if keepAspectRatio and keepSolidity and keepHeight:
            hull =cv2.convexHull(c)
            cv2.drawContours(charCandidates,[hull], -1, 255, -1)
            
        charCandidates = segmentation.clear_border(charCandidates)


cv2.imshow("final", charCandidates)
cv2.imshow("thresh", thresh)
cv2.imshow("image", image)
cv2.imshow("scan", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()