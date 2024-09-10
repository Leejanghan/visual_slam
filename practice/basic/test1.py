#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:07:47 2024

@author: janghan0222
"""

import cv2 
# In Linux --> cv2.CAP_V4L2
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

if not cap.isOpened():
    print('unable to read camera feed')
running = True

while running:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        running = False
        
cap.release()
cv2.destroyAllWindows()
