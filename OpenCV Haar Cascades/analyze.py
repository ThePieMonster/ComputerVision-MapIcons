# Imports
import os
import sys
import time
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

# Analyze Screenshots
print("Analyze Screenshots")
img = cv2.imread('Data/image.png')
classifier_path = 'Data/train-color/classifier/cascade.xml'

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("- load classifier")
cascade = cv2.CascadeClassifier(classifier_path)
print("- detect objects")
objects = cascade.detectMultiScale(image=img_rgb, scaleFactor=1.10, minNeighbors=3)
print("- draw rectangle around objects")
for(x,y,w,h) in objects:
  # cv2.rectangle(<image>, (x,y), (x+w,y+h), <rectangle rgb color>, <rectangle thickness>)
  img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
print()
