import cv2
import numpy as np
from matplotlib import pyplot as plt

# top hat transformation
img2 = cv2.imread('i1.jpg',0)

kernel = np.ones((13,13),np.uint8)
img = cv2.morphologyEx(img2, cv2.MORPH_TOPHAT, kernel)
#cv.imwrite('tophat5.jpg',tophat)







#img = cv2.imread('i1.jpg',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]


# Otsu Threshold image on Original Image
#plt.figure()
#plt.imshow(images[5],'gray')
cv2.imwrite('Otsu1.jpg',images[5])

# Otsu Threshold image after Gaussian filter Image
#plt.figure()
#plt.imshow(images[5],'gray')
cv2.imwrite('Otsu1_Gaussian.jpg',images[8])



#plt.figure()
#for i in range(3):
#    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#plt.show()
