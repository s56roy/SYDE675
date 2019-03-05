#!/usr/bin/env python

'''
This program illustrates the use of findContours and drawContours.
The original image is put up along with the image of drawn contours.

Usage:
    contours.py
A trackbar is put up which controls the contour level from -3 to 3
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv


if __name__ == '__main__':
    print(__doc__)

    img = cv.imread('Otsu1_Gaussian.jpg',0)
    h, w = img.shape[:2]

    _, contours0, hierarchy = cv.findContours( img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

    def update(levels):
        vis = np.zeros((h, w, 3), np.uint8)
        levels = levels - 3
        cv.drawContours( vis, contours, (-1, 2)[levels <= 0], (128,255,255),
            3, cv.LINE_AA, hierarchy, abs(levels) )
        cv.imshow('contours', vis)
    update(3)
    cv.createTrackbar( "levels+3", "contours", 3, 7, update )
    cv.imshow('image', img)
    cv.waitKey()
    cv.destroyAllWindows()
