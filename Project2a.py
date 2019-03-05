import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('Otsu1_Gaussian.jpg',0)



largest_area=0;
largest_contour_index=0

#ret, thresh = cv2.threshold(img2, 40, 255, 0)
#contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
_, contours0, hierarchy = cv2.findContours( img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
#cnt = contours[0]
for cnt in contours0:
    area = cv2.contourArea(cnt)
    if (area>largest_area):
         largest_area=area
        #largest_contour_index=i
	 largest_contour=cnt
        #bounding_rect=cv2.boundingRect(cnt)
#rect=img(bounding_rect).clone()
#rect=img(bounding_rect)
#cv2.imshow('largest contour ',rect)
#cv2.imshow('largest contour ',bounding_rect)
cv2.imshow(larget_contour)
#cv2.waitKey()
#cv2.destroyAllWindows()
